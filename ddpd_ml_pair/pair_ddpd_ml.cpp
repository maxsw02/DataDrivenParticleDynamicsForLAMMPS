// clang-format off
/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 https://www.lammps.org/, Sandia National Laboratories
 LAMMPS development team: developers@lammps.org

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author:
    Max Win (UPENN)
    references: Espanol and Revenga, Phys Rev E 67, 026705 (2003)
------------------------------------------------------------------------- */

#include "pair_ddpd_ml.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "random_park.h"
#include "random_mars.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <omp.h>
//#endif
#include <mpi.h>

// torch 2.6 required for AOT Inductor
#if (TORCH_VERSION_MAJOR < 2 && TORCH_VERSION_MINOR <= 6)
#error "NEQUIP_AOT_COMPILE requires PyTorch >= 2.6"
#endif

using namespace LAMMPS_NS;
using namespace std;

/* ---------------------------------------------------------------------- */
 PairDDPDML::PairDDPDML(LAMMPS *lmp): Pair (lmp)
{ restartinfo = 0;
  manybody_flag = 1;  
  nmax = 0;
  comm_forward = 2;
  d = nullptr;
  nm_Npair=atom->natoms*(atom->natoms+1)/2;


  random = nullptr;

  //single_enable = 0;
  //first_flag = 1;

  if(torch::cuda::is_available()){
    device = torch::kCUDA;
  }
  else {
    device = torch::kCPU;
  }
  if (comm->me == 0){
  std::cout << "Using device " << device << "\n";}

  if (const char *env_p = std::getenv("DEBUG_MODE")) {
    if (comm->me == 0){
    std::cout << "PairDDPDML is in DEBUG mode, since DEBUG is in env\n";}
    debug_mode = 1;
  }
    if (comm->me == 0)
    std::cout << "ddpd/ml is using input precision double " << 
           " and output precision float" << std::endl;
}

/* ---------------------------------------------------------------------- */

 PairDDPDML::~PairDDPDML() {
  if (copymode) return;
  memory->destroy(d);
  if (allocated) {
    memory->destroy (setflag);
    memory->destroy (cutsq);
    memory->destroy (cut);

  };
  memory->destroy(pair_check);
  if (random) delete random;



}

/* ---------------------------------------------------------------------- */

 void PairDDPDML::compute (int eflag, int vflag) {

  ev_init(eflag, vflag);
  int i, j , jnum, itype, jtype;
  int *numneigh, *jlist, **firstneigh;
  double wiener[3][3],sym_wiener[3][3], f_random[3], f_rand[3];
  double frandx, frandy, frandz;
  double xtmp, ytmp, ztmp, delx, dely, delz, ex,ey,ez, rsq, r, dV;
  double trace, vx, vy, vz;
  double a_loc, b_loc, c_loc;
  double term_rand1, term_rand2;
  double vrand_const, srand_const;
  
  double **x = atom->x;
  double dtinv = 1.0 / update->dt;
  double sqrtdtinv = sqrt(dtinv);
  double **v = atom->v;
  int *type = atom->type;
  int newton_pair = force->newton_pair;
  int nlocal = atom->nlocal;

  // Atom forces
  double **f = atom->f;
  double *dentropy = atom->dentropy;

  int inum = list->inum;
  if (inum==0) return;

  // Number of ghost atoms
  int nghost = list->gnum;
  // Total number of atoms
  int ntotal = inum + nghost;

  // Mapping from neigh list ordering to x/f ordering
  // (in case we want to support pair_coeff other than * * in the future)
  int *ilist = list->ilist;
  int *tag = atom->tag;
  int itag, jtag;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  //input.clear();
  // create input to model (positions etc)

  c10::Dict<std::string, torch::Tensor> model_input = preprocess();
  std::vector<torch::jit::IValue> input_vector(1, model_input);
  auto output = model.forward(input_vector).toGenericDict();

  torch::Tensor f_tensor = output.at("dvdt").toTensor().cpu().detach();
  //torch::Tensor f_tensor = output[0].cpu();
  auto forces = f_tensor.accessor<float, 2>();
  //torch::Tensor dS_tensor = output[1].cpu();
  torch::Tensor dS_tensor = output.at("dSdt").toTensor().cpu().detach();
  auto dS = dS_tensor.accessor<float,2>();
  //torch::Tensor e_tensor = output[2].cpu();
  torch::Tensor e_tensor = output.at("E").toTensor().cpu().detach();
  auto atomic_energies = e_tensor.accessor<float, 2>();


  if(debug_mode && comm->me == 0){
    std::cout << "Model output:\n";

    std::cout << "dvdt: " << f_tensor << "\n";
    std::cout << "dSdt: " << dS_tensor << "\n";
    std::cout << "E: " << e_tensor << "\n";
  }
  eng_vdwl = 0.0;
  #pragma omp parallel for reduction(+ : eng_vdwl)
    for (int ii = 0; ii < inum; ii++) {
      i = ilist[ii];

      f[i][0] += forces[i][0]; 
      f[i][1] += forces[i][1];
      f[i][2] += forces[i][2];
      dentropy[i] += dS[i][0];
      if (eflag_atom && ii < inum) eatom[i] = atomic_energies[i][0];
      if (ii < inum) eng_vdwl += atomic_energies[i][0];}
}



/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

 void PairDDPDML::allocate () {
  allocated = 1;
  int n = atom->ntypes;
  //std::cout << "ntypes = " << n << "\n";

  memory->create (setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create (cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create (cut, n+1 , n+1 , "pair:cut");

  memory->create(pair_check, nm_Npair,"pair:pair_check");
  for(int i=0; i<nm_Npair; i++)
        pair_check[i]=false;
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

 void PairDDPDML::settings (int narg, char **arg) {
  //std::cout << narg << "\n";
  if (narg != 1) error->all(FLERR, "Illegal pair_style command, too many arguments");

  int seed = utils::inumeric(FLERR, arg[0], false, lmp);
  if (seed <= 0) error->all(FLERR,"Invalid random number seed");

  delete random;

  random = new RanMars(lmp,(seed + comm->me) % 900000000);
  int natoms = atom->natoms;
}
/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

 void PairDDPDML::coeff (int narg, char **arg) {
  
  if (!allocated) allocate();
  int ntypes = atom->ntypes;

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 0;

  if (narg != 5) {
    error->all(FLERR,
	        "Incorrect args for pair coefficients, need 5 args");
  }

  // Ensure I,J args are "* *".
  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double cut_one = utils::numeric(FLERR,arg[2], false, lmp);

  self_cut = cut_one;
  if (cut_one <= 0) error->all (FLERR, "Cutoff must be positive");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      if (comm->me == 0) printf("setting cut[%d][%d] = %f\n", i, j, cut_one);
      cut[i][j] = cut_one;
      cutsq[i][j] = cut_one * cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0 && comm->me == 0){error->all(FLERR,"Incorrect args for pair coefficients");}

  model = torch::jit::load(std::string(arg[3]), device);
  modelW = torch::jit::load(std::string(arg[4]), device);
  model.eval();
  modelW.eval();
  if (comm->me == 0) {std::cout << "DDPD/ML: Freezing TorchScript model... "<<std::endl;
  std::cout << "DDPD/ML: Freezing TorchScript model W..." <<std::endl;
  }
  model = torch::jit::freeze(model);
  modelW = torch::jit::freeze(modelW);
  torch::jit::FusionStrategy strategy = {{torch::jit::FusionBehavior::DYNAMIC, 5}};
  torch::jit::setFusionStrategy(strategy);

  //int omp_threads = omp_get_max_threads();
  //torch::set_num_threads(omp_threads);
  //if (comm->me == 0) printf("Using %d threads for PyTorch\n", omp_threads);
}

/* ----------------------------------------------------------------------
 init specific to this pair style
------------------------------------------------------------------------ */
 void PairDDPDML::init_style()
{
  first_flag = 0;

  if ((!atom->dentropy_flag) || (atom->dentropy == nullptr))
    error->all(FLERR,"Pair style ddpd/ml requires atom attributes entropy and dentropy");

  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  
  if (atom->tag_enable == 0) error->all(FLERR,"Pair style ddpd/ml requires atom IDs");
} 

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

 double PairDDPDML::init_one (int i, int j) {
  if (setflag[i][j] == 0)
    error->all(FLERR,"Not all pair ddpd/ml coeffs are set");
  return self_cut;
}

 c10::Dict<std::string, torch::Tensor> PairDDPDML::preprocess() {
  double trace;
  // Atom positions, including ghost atoms
  double **x = atom->x;
  // Atom IDs, unique, reproducible, the "real" indices
  // Probably 1-based
  tagint *tag = atom->tag;
  // Atom types, 1-based
  int *type = atom->type;
  // Number of local/real atoms
  int nlocal = atom->nlocal;

  // Number of local/real atoms
  int inum = list->inum;
  assert(inum == nlocal);    // This should be true, if my understanding is correct
  // Number of ghost atoms
  int nghost = list->gnum;
  // Total number of atoms
  //int ntotal = inum;
  int ntotal = inum + nghost;
  int newton_pair = force->newton_pair;

  // Mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // Number of neighbors per atom
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  int **firstneigh = list->firstneigh;

  double **v = atom->v;
  double *entropy = atom->entropy;

  if (atom->nmax > nmax) {
    memory->destroy(d);
    nmax = atom->nmax;
    memory->create(d,nmax,"pair:d");
  }

std::vector<int> neigh_per_atom(inum, 0); //parser number of edges for d and S
int nedges = 0; //local number of edges for d and S
//#pragma omp parallel
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int itag = tag[i];

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      int jtag = tag[j];

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx * dx + dy * dy + dz * dz;
      double cutij =
          cut[type[i] ][type[j]];
      if (rsq <= cutij * cutij) {
        if (j > i){
        neigh_per_atom[ii]++;
        nedges++;
        }
      }
    }
  }  
  cumsum_neigh_per_atom.clear();
  cumsum_neigh_per_atom.resize(inum,0);
  for (int ii = 1; ii < inum; ii++) {
    cumsum_neigh_per_atom[ii] = cumsum_neigh_per_atom[ii - 1] + neigh_per_atom[ii - 1];
  }


  if (debug_mode){ std::cout << "nedges: " << nedges  << "\n";}

  torch::Tensor v_tensor = torch::zeros({ntotal, 3}, device);
  torch::Tensor edges_tensor = torch::zeros({2, nedges}, torch::TensorOptions().dtype(torch::kInt64)).to(device);
  torch::Tensor r_ij_tensor = torch::zeros({nedges, 3}, device);
  torch::Tensor dW_ij = torch::zeros({nedges,3,3}, device);
  torch::Tensor dV_ij = torch::zeros({nedges,1}, device);
  auto dw = dW_ij.accessor<float, 3>();
  auto dV = dV_ij.accessor<float, 2>();
  

  auto vel = v_tensor.accessor<float, 2>();
  auto edges = edges_tensor.accessor<long, 2>();
  auto r_ij = r_ij_tensor.accessor<float, 2>();

  torch::Tensor S_2  = torch::ones({ntotal,1}, device);
  torch::Tensor d_tot = torch::zeros({ntotal,1}, device);

  auto s2 = S_2.accessor<float, 2>();
  auto d_all_acc = d_tot.accessor<float, 2>();
 
 //Loop over atoms and neighbors,
 //store edges and velocities
//#pragma omp parallel for if(!debug_mode)
int edge_counter = 0;
//int pair_index= 0;
  for (int ii = 0; ii < ntotal; ii++) {
    int i = ilist[ii];
    int itag = tag[i];
    int itype = type[i];
    
    vel[i][0] = v[i][0];
    vel[i][1] = v[i][1];
    vel[i][2] = v[i][2];

    s2[i][0] = entropy[i];
    if (ii >= inum) {continue;}    
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    int edge_counter = cumsum_neigh_per_atom[ii];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      int jtag = tag[j];
      int jtype = type[j];

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];
      double rsq = dx * dx + dy * dy + dz * dz;
      double cutij =cut[itype][jtype];
      if (rsq < cutij * cutij) { 
      if ( j > i){
        r_ij[edge_counter][0] = dx;
        r_ij[edge_counter][1] = dy;
        r_ij[edge_counter][2] = dz;
      
        edges[0][edge_counter] = i;
        edges[1][edge_counter] = j;

        dw[edge_counter][0][0] = random->gaussian(0.0,1.0);
        dw[edge_counter][1][1] = random->gaussian(0.0,1.0);
        dw[edge_counter][2][2] = random->gaussian(0.0,1.0);
        dw[edge_counter][0][1] = random->gaussian(0.0,1.0);
        dw[edge_counter][0][2] = random->gaussian(0.0,1.0);
        dw[edge_counter][1][2] = random->gaussian(0.0,1.0);
        dw[edge_counter][2][0] = random->gaussian(0.0,1.0);
        dw[edge_counter][2][1] = random->gaussian(0.0,1.0);
        dw[edge_counter][1][0] = random->gaussian(0.0,1.0);
        dV[edge_counter][0] = random->gaussian(0.0,1.0);

        edge_counter++;
      }
      }
    }
  }
  
  torch::Tensor N_tensor = torch::tensor({ntotal}, torch::kInt64).to(device); //total number of atoms
  c10::Dict<std::string, torch::Tensor> inputW;
  inputW.insert("r_ij", r_ij_tensor);
  inputW.insert("edge_index", edges_tensor);
  inputW.insert("N", N_tensor);
  std::vector<torch::IValue> inputW_vector(1, inputW);

  torch::Tensor d_tensor = modelW.forward(inputW_vector).toTensor();
  auto d_acc= d_tensor.accessor<float, 2>();

//#pragma omp parallel
  for (int ii = 0; ii < inum; ii++) {
      int i = ilist[ii];
      d[i] = d_acc[i][0];
    }
comm->forward_comm(this); //Communicate d and S to all atoms

//#pragma omp parallel
  for (int ii = 0; ii < ntotal; ii++) {
      int i = ilist[ii];
      d_all_acc[i][0] = d[i];
      s2[i][0] = entropy[i];
    }

c10::Dict<std::string, torch::Tensor> model_inputs;
model_inputs.insert("v", v_tensor);
model_inputs.insert("edge_index", edges_tensor);
model_inputs.insert("r_ij", r_ij_tensor);
model_inputs.insert("S", S_2);
model_inputs.insert("d", d_tot);
model_inputs.insert("dW", dW_ij);
model_inputs.insert("dV", dV_ij);

if(debug_mode && comm->me == 0){
  std::cout << "model input:\n";
  std::cout << "v " << v_tensor << std::endl;
  std::cout << "S " << S_2 << std::endl;
  std::cout << "edge_index:\n" << edges_tensor << std::endl;
  std::cout << "r_ij:\n" << r_ij_tensor << std::endl;
  std::cout << "d:\n" << d_tot << std::endl;
  std::cout << "edge_index size: " << edges_tensor.sizes() << "\n";
  }
  return model_inputs;
}

void *PairDDPDML::extract_peratom(const char *str, int &ncol)
{
  if (strcmp(str,"d") == 0) {
    ncol = 0;
    return (void *) d;
  }

  return nullptr;
}

int PairDDPDML::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,m;  
  double *entropy = atom->entropy;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = entropy[j];
    buf[m++] = d[j];
  }
  return m;
}

void PairDDPDML::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;
  double *entropy = atom->entropy;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++) {
    entropy[i] = buf[m++];
    d[i] = buf[m++];}
}

double PairDDPDML::memory_usage()
{
  int n = atom->ntypes;
  int nall = atom->natoms;
  double bytes = 2*(double)nmax * sizeof(double);
  bytes += (n + 1) * (n + 1) * sizeof(int);     // setflag
  bytes += (n + 1) * (n + 1) * sizeof(double);  // cutsq
  bytes += (n + 1) * (n + 1) * sizeof(double);
  bytes += 0.5 * nall * (nall+1) * sizeof(double);
  bytes += 2 * 3 * 3 * sizeof(double);
  bytes += 3  * sizeof(double);
  return bytes;
}
