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

def generate_lammps_input(nparticles ):
    input = r"""variable        scaling equal 0.03637829108 #particle number density \n
variable        h equal 4.75    
variable        mass equal 1.0 
variable        dt equal 0.025  
variable        s_per equal 0.1 
variable        nparticles equal {}
variable        L equal (v_nparticles/v_scaling)^(1./3.)
units lj              
atom_style sph     
dimension 3           
boundary p p p       
region box block 0 ${{L}} 0 ${{L}} 0 ${{L}}  
create_box 1 box                  
create_atoms 1 random ${{nparticles}} 12345 box overlap 2.2 
mass            * ${{mass}} 
velocity        all create 50000. {} loop local dist gaussian #defined from training simulations
write_data      atoms_big.data""".format(nparticles, np.random.randint(0, high=100000))

    with open("generate_data.lammps", "w") as f:
        f.write(input)
    os.system("/home/max/Desktop/lammps_sdpd_rand_only/jit_opt_build/lmp -in generate_data.lammps")

    return 0

def generate_exec_files():
    input1=r"""variable        h equal 4.75      # pairstyle cutoff determined during ML training   
variable        dt equal 0.025  # timestep
variable 	eq_steps equal 9600
units lj              # Lennard-Jones units (dimensionless)
atom_style sph     # Basic atomic style for particles
dimension 3           # 3D simulation
boundary p p p        # Periodic boundaries in all directions


newton on
read_data atoms_mod.data


compute          ent all property/atom entropy
compute          dent all property/atom dentropy
compute          entropy all reduce sum c_ent
compute mom all momentum

variable natoms equal atoms
pair_style      sdpd/ml {}
pair_coeff      * * ${{h}} sdpd_jit_rand.pt sdpd_jit_w.pt
fix              1 all sph/sdpd

timestep         ${{dt}}

comm_modify  vel yes #communicate velocities
thermo 10
thermo_style  custom step temp pe ke etotal press c_entropy c_mom[1] c_mom[2] c_mom[3]
dump    f       all custom 5 output_1n_f4.xyz id x y z xu yu zu vx vy vz c_ent c_dent
run 400
undump f

run             ${{eq_steps}}


dump    1       all custom 5 output_1n.xyz id x y z xu yu zu vx vy vz c_ent c_dent
dump_modify 1 sort id


run  1000

undump 1""".format(np.random.randint(0, high=10000)) 

    with open("sdpd_exec.in", "w") as f:
        f.write(input1)


def write_lammps_data_from_numpy(filename, box_bounds, entropy, position, velocity, atom_id, atom_type,
                                  default_rho=0.0, default_esph=0.0, default_cv=1.0, timestep=0, units='lj'):
    N = len(atom_id)
    assert position.shape == (N, 3)
    assert velocity.shape == (N, 3)
    assert entropy.shape == (N,)
    assert atom_type.shape == (N,)
    
    with open(filename, 'w') as f:
        f.write(f"LAMMPS data file via write_data, version 2 Aug 2023, timestep = {timestep}, units = {units}\n\n")
        f.write(f"{N} atoms\n")
        f.write(f"{np.unique(atom_type).size} atom types\n\n")

        for dim in ['x', 'y', 'z']:
            lo, hi = box_bounds[dim]
            f.write(f"{lo} {hi} {dim}lo {dim}hi\n")
        f.write("\nMasses\n\n1 1\n\n")

        f.write("Atoms # sph\n")
        f.write("#atom-id atom-type rho esph cv entropy x y z ix iy iz\n")
        for i in range(N):
            f.write(f"{atom_id[i]} {atom_type[i]} {default_rho} {default_esph} {default_cv} {entropy[i]} "
                    f"{position[i,0]} {position[i,1]} {position[i,2]} 0 0 0\n")

        f.write("\nVelocities\n\n")
        for i in range(N):
            f.write(f"{atom_id[i]} {velocity[i,0]} {velocity[i,1]} {velocity[i,2]}\n")


if __name__ == '__main__':
    model_S_jit = torch.jit.load("sdpd_jit_s.pt")
    generate_lammps_input(1000)
    u = mda.Universe('atoms_big.data', atom_style='id type rho esph cv entropy x y z')
    dims = u.dimensions
    h = 4.75
    edge_index, r_ij = get_edges(u.atoms.positions, dims[:3],h)
    inputs_S = {'edge_index': torch.tensor(edge_index), 'r_ij': torch.tensor(r_ij, dtype=torch.int64), 'v': torch.tensor(u.atoms.velocities)}
    s = model_S_jit(inputs_S)

    box_bounds = {'x': (0.0, dims[0]), 'y': (0.0, dims[1]), 'z': (0.0, dims[2])}
    entropy = s.detach().numpy().squeeze()
    position = u.atoms.positions
    velocity = u.atoms.velocities

    atom_id = u.atoms.ids
    atom_type = u.atoms.types
    write_lammps_data_from_numpy("atoms_mod.data", box_bounds, entropy, position, velocity, atom_id, atom_type)
    generate_exec_files()
