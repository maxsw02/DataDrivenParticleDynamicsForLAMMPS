/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(ddpd/ml,PairDDPDML);
#else

#ifndef LMP_PAIR_DDPD_ML
#define LMP_PAIR_DDPD_ML

#include "pair.h"
#include <torch/torch.h>
#include <vector>
#include <type_traits>
#include <map>
#include <random>
#include <string>


//#endif
//enum Precision {lowlow, highhigh, lowhigh, highlow};

namespace LAMMPS_NS {

//template<Precision precision>
class PairDDPDML : public Pair {
 public:
  int first_flag;
  PairDDPDML(class LAMMPS *);
  ~PairDDPDML() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  void allocate();
  void *extract_peratom(const char *, int &) override;
  //int pack_reverse_comm(int, int, double *) override;
  //void unpack_reverse_comm(int, int *, double *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double memory_usage() override;

  std::string model_path;
  std::string model_W_path;

  torch::Device device = torch::kCPU;

  torch::jit::Module model;
  torch::jit::Module modelW;

 protected:
  double **cut;
  double cutforcesq, self_cut;
  int nmax;
  int nm_Npair;
  // Per-atom array
  double *d = nullptr;
  bool *pair_check;

   
   unsigned int seed;
   class RanMars *random;
   int debug_mode = 0;
   std::vector<int> cumsum_neigh_per_atom;
   c10::Dict<std::string, torch::Tensor> preprocess();
};

}    // namespace LAMMPS_NS
#endif
#endif