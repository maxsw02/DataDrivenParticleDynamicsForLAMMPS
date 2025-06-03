
import os
import numpy as np
import argparse

import torch
from tqdm import tqdm
import json

from .generate_lammps import get_edges, write_lammps_data_from_numpy, generate_exec_files

from .model import CG_model
from .model_jit import CG_model_jit
from .model_jit import CG_model_S_jit
from .model_jit import CG_model_W_jit

import MDAnalysis as mda


class JITCompile(object):
    def __init__(self, args, params_dir,lmp_dir, dims):
        self.args = args
        self.params_file = params_dir
        self.output_dir = lmp_dir

        # Study Case
        #self.device = torch
        self.device = torch.device('cpu') #will do cuda later because LAMMPS kokkos pair style not yet implemented

        # Dataset Parameters
        self.dt = args.dt
        self.h = args.h
        self.dims = dims

        # Net Parameters
        self.model = CG_model(args, self.dims).to(self.device).float()
        checkpoint = torch.load(self.params_file, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint)


    def export_jit(self):
        # Load parameters to JIT class
        #load_path = os.path.join(self.data_dir, 'params.pt')
        checkpoint = torch.load(self.params_file, map_location=self.device, weights_only=False)

        self.model_S_jit = CG_model_S_jit(self.args, self.dims).to(self.device).float()
        self.model_S_jit.load_state_dict({k: v for k, v in checkpoint.items() if k.startswith('teacher.')})

        self.model_W_jit = CG_model_W_jit(self.args, self.dims).to(self.device).float()
        self.model_W_jit.load_state_dict({k: v for k, v in checkpoint.items() if k.startswith('model_W.')})

        self.model_jit = CG_model_jit(self.args, self.dims).to(self.device).float()
        self.model_jit.load_state_dict({k: v for k, v in checkpoint.items() if not k.startswith('teacher.')})

        # Check I/O
        inputs = {'r_ij': torch.rand((1000,3), device=self.device),
                  'v': torch.rand((100,3), device=self.device),
                  'edge_index': torch.randint(0,100,(2,1000), device=self.device)}
        _ = self.model_S_jit(inputs)

        inputs = {'r_ij': torch.rand((1000,3), device=self.device),
                  'edge_index': torch.randint(0,100,(2,1000), device=self.device),
                  'N': 100 * torch.ones((1), device=self.device, dtype=torch.int64)}
        _ = self.model_W_jit(inputs)

        inputs = {'v': torch.rand((1000,3), device=self.device), 
                  'S': torch.rand((1000,1), device=self.device), 
                  'edge_index': torch.randint(0,3,(2,10000), device=self.device),
                  'r_ij': torch.rand((10000,3), device=self.device),
                  'd': torch.rand((1000,1), device=self.device),
                  'dW': torch.rand((10000,3,3), device=self.device),
                  'dV': torch.rand((10000,1), device=self.device)}
        _ = self.model_jit(inputs)

        # Save JIT network
        save_dir = os.path.join(self.output_dir, 'params_S_jit.pt')
        torch.jit.script(self.model_S_jit).save(save_dir)
        save_dir = os.path.join(self.output_dir, 'params_W_jit.pt')
        torch.jit.script(self.model_W_jit).save(save_dir)
        save_dir = os.path.join(self.output_dir, 'params_jit.pt')
        torch.jit.script(self.model_jit).save(save_dir)

    def calc_entropies(self, S_model, lammps_data_file_input , cutoff, output_data_file):
        u = mda.Universe(lammps_data_file_input, atom_style='id type rho esph cv entropy x y z')
        dims = u.dimensions
        edge_index, r_ij = get_edges(u.atoms.positions, dims[:3],cutoff)
        inputs_S = {'edge_index': torch.tensor(edge_index, device= self.device), 'r_ij': torch.tensor(r_ij, dtype=torch.int64, device= self.device), 'v': torch.tensor(u.atoms.velocities, device= self.device)}
        print("Entropy Calculation used {} device".format(self.device))

        s = S_model(inputs_S)

        if self.device == torch.device('cuda'):
            entropy = s.cpu().data.numpy().squeeze()
        elif self.device == torch.device('cpu'):
            entropy = s.detach().numpy().squeeze()
        
        position = u.atoms.positions
        velocity = u.atoms.velocities
        atom_id = u.atoms.ids
        atom_type = u.atoms.types
        unique_types = np.unique(atom_type)
        box_bounds = {'x': (0.0, dims[0]), 'y': (0.0, dims[1]), 'z': (0.0, dims[2])}
        write_lammps_data_from_numpy(output_data_file, box_bounds, entropy, position, velocity, atom_id, atom_type, unique_types)

def json_load(path):
    with open(path, 'r') as f:
        return json.load(f)




