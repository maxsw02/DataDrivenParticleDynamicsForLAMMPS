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

#include "atom_vec_sph.h"

#include "atom.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecSPH::AtomVecSPH(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = Atom::ATOMIC;
  mass_type = PER_TYPE;
  forceclearflag = 1;
  
  atom->esph_flag = 1;
  atom->rho_flag = 1;
  atom->cv_flag = 1;
  atom->vest_flag = 1;
  atom->entropy_flag = 1;
  atom->dentropy_flag = 1;
  //atom->temperature_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = {"rho", "drho", "esph", "desph", "cv", "vest", "entropy", "dentropy"};
  fields_copy = {"rho", "drho", "esph", "desph", "cv", "vest", "entropy", "dentropy"};
  fields_comm = {"rho", "esph", "vest", "entropy", "dentropy"};
  fields_comm_vel = {"rho", "esph", "vest","entropy", "dentropy"};
  fields_reverse = {"drho", "desph","dentropy"};
  fields_border = {"rho", "esph", "cv", "vest","entropy"};
  fields_border_vel = {"rho", "esph", "cv", "vest","entropy"};
  fields_exchange = {"rho", "esph", "cv", "vest","entropy"};
  fields_restart = {"rho", "esph", "cv", "vest","entropy"};
  fields_create = {"rho", "esph", "cv", "vest", "desph", "drho", "entropy", "dentropy"};
  fields_data_atom = {"id", "type", "rho", "esph", "cv","entropy", "x"};
  fields_data_vel = {"id", "v"};


  setup_fields();
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecSPH::grow_pointers()
{
  rho = atom->rho;
  drho = atom->drho;
  esph = atom->esph;
  desph = atom->desph;
  cv = atom->cv;
  vest = atom->vest;
  entropy = atom->entropy;
  dentropy = atom->dentropy;
  //temperature = atom->temperature;
}

/* ----------------------------------------------------------------------
   clear extra forces starting at atom N
   nbytes = # of bytes to clear for a per-atom vector
------------------------------------------------------------------------- */

void AtomVecSPH::force_clear(int n, size_t nbytes)
{
  memset(&desph[n], 0, nbytes);
  memset(&drho[n], 0, nbytes);
  memset(&dentropy[n], 0, nbytes);
}

/* ----------------------------------------------------------------------
   initialize non-zero atom quantities
------------------------------------------------------------------------- */

void AtomVecSPH::create_atom_post(int ilocal)
{
  cv[ilocal] = 1.0;
  entropy[ilocal] = 1.0;
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecSPH::data_atom_post(int ilocal)
{
  vest[ilocal][0] = 0.0;
  vest[ilocal][1] = 0.0;
  vest[ilocal][2] = 0.0;
  desph[ilocal] = 0.0;
  drho[ilocal] = 0.0;

  dentropy[ilocal] = 0.0;
  //temperature[ilocal] = 0.0;
}

/* ----------------------------------------------------------------------
   assign an index to named atom property and return index
   return -1 if name is unknown to this atom style
------------------------------------------------------------------------- */

int AtomVecSPH::property_atom(const std::string &name)
{
  if (name == "rho") return 0;
  if (name == "drho") return 1;
  if (name == "esph") return 2;
  if (name == "desph") return 3;
  if (name == "cv") return 4;
  if (name == "dentropy") return 5;
  if (name == "entropy") return 6;
  //if (name == "temperature") return 7;
  return -1;
}

/* ----------------------------------------------------------------------
   pack per-atom data into buf for ComputePropertyAtom
   index maps to data specific to this atom style
------------------------------------------------------------------------- */

void AtomVecSPH::pack_property_atom(int index, double *buf, int nvalues, int groupbit)
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int n = 0;

  if (index == 0) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = rho[i];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 1) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = drho[i];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 2) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = esph[i];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 3) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = desph[i];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 4) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = cv[i];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 5) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = dentropy[i];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 6) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = entropy[i];
      else
        buf[n] = 0.0;
      n += nvalues;
    } 
    }
}

