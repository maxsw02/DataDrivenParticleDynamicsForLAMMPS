import os
import torch
import argparse

from utils.jitCompile import JITCompile, json_load
from utils.generate_lammps import generate_exec_files 
from argparse import Namespace


def main(args):
    #Make LAMMPS simulation directory
    cwd = os.getcwd()
    metadata = json_load(args.metadata)
    metadata = Namespace(**metadata)

    lammps_path = os.path.join(cwd,args.lammps_sim_dir)
    lammps_data_file_pre_entropy = os.path.join(cwd,args.lmp_setup_data)
    lammps_data_file_entropy = os.path.join(lammps_path,args.lmp_entropy_data)
    lammps_log_init_config_file = os.path.join(lammps_path,'log_init_config.lammps')

    if not os.path.exists(lammps_path):
        os.makedirs(lammps_path, exist_ok=True)
    os.system("cp {} {}".format(args.lammps_setup, lammps_path))
    print("LAMMPS is generating pre-entropy data file")
    os.system("{} -in {} > {}".format(args.lammps, args.lammps_setup, lammps_log_init_config_file))
    os.remove("log.lammps")
    print("LAMMPS is finished generating pre-entropy data file")
    os.system("mv {} {}".format(lammps_data_file_pre_entropy, lammps_path))

    #os.system("mv pre_entropy.data {}".format(lammps_data_file_pre_entropy))

    #Save jitted models to LAMMPS simulation directory
    
    jit_model =JITCompile(metadata,args.params, lammps_path,args.dims)
    jit_model.export_jit()

    #Calculate initial entropies from LAMMPS generated data file
    S_jit_model_path = os.path.join(lammps_path,'params_S_jit.pt')
    S_model =torch.jit.load(S_jit_model_path)
    new_loc_pre_entropy_data_file = os.path.join(lammps_path, args.lmp_setup_data)
    jit_model.calc_entropies(S_model, new_loc_pre_entropy_data_file, metadata.h, lammps_data_file_entropy)

    #Generates a lammps input that reads in data file with entropies and runs the simulation 
    #from the configuation of the data file. It will use the timestep and cutoff radius from 
    # the training data.
    if args.lmp_custom_input:
        print("copying custom exec file, {}, to lammps directory".format(args.lmp_custom_input))
        os.system("cp {} {}".format(args.lmp_custom_input, lammps_path))
    elif args.lmp_custom_input == None:
        generate_exec_files(metadata.h, metadata.dt, lammps_data_file_entropy, lammps_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JIT Model Compilation')

    #LAMMPS stuff
    parser.add_argument('--lammps', default=' mpirun -np 4 /home/max/Desktop/LAMMPS/src/lmp_mpi ', type=str, help='LAMMPS executable') #LAMMPS executable 
    parser.add_argument('--lammps_setup', default='setup.in', type=str, help='LAMMPS setup script') #Generate initial conditions/geometry write a data file with positions and velocities
    parser.add_argument('--lammps_sim_dir', default='lmp_sim', type=str, help='LAMMPS simulation directory')
    parser.add_argument('--lmp_setup_data', default='pre_entropy.data', type=str, help='name of LAMMPS data file of user defined initial conditions') #This will be produced by this executable
    parser.add_argument('--lmp_entropy_data', default='lmp_entropy.data', type=str, help='name of LAMMPS data file of user defined initial conditions with calculated entropies')#This will be produced by this executable
    parser.add_argument('--lmp_custom_input', default=None, type=str, help='If user wants to run the simulaton with a their own LAMMPS parameters, then sdpd_exec.in file will not be generated')

    #Metadata from training i.e. cutoff radius, timestep, etc.
    parser.add_argument('--metadata', default='args.json', type=str, help='location of metadata file')
    parser.add_argument('--params', default='params.pt', type=str, help='location of model parameters')
    parser.add_argument('--dims', default=3, type=int, help='dimensionality of problem')

    args = parser.parse_args()
    main(args)
