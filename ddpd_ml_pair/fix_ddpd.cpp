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

#include "fix_ddpd.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include <iostream>


using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixDDPD::FixDDPD(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg) {

  if ((atom->esph_flag != 1) || (atom->rho_flag != 1))
    error->all(FLERR,
        "Fix sph command requires atom_style with both energy and density");

  if (narg != 3)
    error->all(FLERR,"Illegal number of arguments for fix sph command");

  time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixDDPD::setmask() {
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDDPD::init() {
  dtv = update->dt;
  dtf = dtv * force->ftm2v; 
  //dtv = update->dt;
  //dtf = 0.5 * update->dt * force->ftm2v;
}

/* ----------------------------------------------------------------------
 allow for both per-type and per-atom mass
 ------------------------------------------------------------------------- */

void FixDDPD::initial_integrate(int /*vflag*/) {
  // update v and x and rho and e of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  double *entropy = atom->entropy;
  double *dentropy = atom->dentropy;
  int rmass_flag = atom->rmass_flag;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int i;
  double dtfm;

  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;

  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (rmass_flag) {
        dtfm = dtf;
        //dtfm = dtf / rmass[i];
      } else {
        dtfm = dtf;
        //dtfm = dtf / mass[type[i]];
      }

      v[i][0] +=  dtfm * f[i][0];
      v[i][1] +=  dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];

      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];

      entropy[i] += dtv * dentropy[i];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixDDPD::final_integrate() {

}

/* ---------------------------------------------------------------------- */

void FixDDPD::reset_dt() {
  dtv = update->dt;
  dtf = dtv * force->ftm2v;
  std::cout << "dtf: " << dtf << "\n";
  std::cout << "dtv: " << dtv << "\n";
}
