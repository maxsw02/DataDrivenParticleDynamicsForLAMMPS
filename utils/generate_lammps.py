import torch
from scipy.spatial import KDTree
import MDAnalysis as mda
import numpy as np
import os
import matplotlib.pyplot as plt

def get_edges(pos, box,h):
    tree = KDTree(pos, boxsize=box)
    edge_index = tree.query_pairs(h, output_type='ndarray').T
    r_ij = (pos[edge_index[0]] - pos[edge_index[1]]) - box * np.round((pos[edge_index[0]] - pos[edge_index[1]]) / box)
    return edge_index, r_ij


def generate_exec_files(cutoff, timestep, data_file, output_dir):
    input1=r"""variable        h equal {}     # pairstyle cutoff determined during ML training   
variable        dt equal {}  # timestep
variable 	eq_steps equal 10000 #equilibriation steps
units lj              # Lennard-Jones units (dimensionless)
atom_style sph     # Basic atomic style for particles
dimension 3           # 3D simulation
boundary p p p        # Periodic boundaries in all directions
newton on
read_data {}
compute          ent all property/atom entropy #calculate entropy_per_particle
compute          dent all property/atom dentropy #calculate dentropy
compute          entropy all reduce sum c_ent #calculate total entropy

variable natoms equal atoms
pair_style      ddpd/ml {}
pair_coeff      * * ${{h}} params_jit.pt params_W_jit.pt 
fix              1 all sph/ddpd

timestep         ${{dt}}

comm_modify  vel yes #communicate velocities (NECCESSARY)
thermo 10
thermo_style  custom step temp pe ke etotal press c_entropy 
run             ${{eq_steps}}

dump    1       all custom 5 output_1n.xyz id x y z xu yu zu vx vy vz c_ent c_dent
#dump_modify 1 sort id
run  1000
undump 1""".format(cutoff, timestep , data_file,np.random.randint(0, high=10000)) 
    path = os.path.join(output_dir,'ddpd_exec.in')
    with open(path, "w") as f:
        f.write(input1)


def write_lammps_data_from_numpy(filename, box_bounds, entropy, position, velocity, atom_id, atom_type,unique_types,
                                  default_rho=0.0, default_esph=0.0, default_cv=1.0, timestep=0, units='lj'):
    N = len(atom_id)
    assert position.shape == (N, 3)
    assert velocity.shape == (N, 3)
    assert entropy.shape == (N,)
    #assert atom_type.shape == (N,)
    
    with open(filename, 'w') as f:
        f.write(f"LAMMPS data file via write_data, version 2 Aug 2023, timestep = {timestep}, units = {units}\n\n")
        f.write(f"{N} atoms\n")
        f.write(f"{np.unique(atom_type).size} atom types\n\n")

        for dim in ['x', 'y', 'z']:
            lo, hi = box_bounds[dim]
            f.write(f"{lo} {hi} {dim}lo {dim}hi\n")
        f.write("\nMasses\n\n")
        for i in unique_types:
            f.write("{} 1.\n".format(i))
        f.write("\nAtoms # sph\n")
        f.write("#atom-id atom-type rho esph cv entropy x y z ix iy iz\n")

        for i in range(N):
            f.write(f"{atom_id[i]} {atom_type[i]} {default_rho} {default_esph} {default_cv} {entropy[i]} "
                    f"{position[i,0]} {position[i,1]} {position[i,2]} 0 0 0\n")
        f.write("\nVelocities\n\n")
        for i in range(N):
            f.write(f"{atom_id[i]} {velocity[i,0]} {velocity[i,1]} {velocity[i,2]}\n")



