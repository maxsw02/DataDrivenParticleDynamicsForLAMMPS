from model import CG_model
from model_jit_w import CG_model_jit as CG_model_jit_w
from model_jit_s import CG_model_jit as CG_model_jit_s
from model_jit_rand import CG_model_jit as CG_model_jit_rand
from sys import argv

import torch

def str2bool(v):
    # Code from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import argparse
parser = argparse.ArgumentParser(description='Machine Learning of Smoothed Dissipative Dynamics')
parser.add_argument('--gpu', default=True, type=str2bool, help='GPU acceleration')
parser.add_argument('--dt', default=0.025, type=float, help='time step')
parser.add_argument('--h', default=4.75, type=float, help='cutoff radius')
parser.add_argument('--boxsize', default=30.18, type=float, help='length of periodic box or None')
parser.add_argument('--n_hidden', default=2, type=int, help='number of hidden layers per MLP')
parser.add_argument('--dim_hidden', default=100, type=int, help='dimension of hidden units')
parser.add_argument('--m', default=1.0, type=float, help='mass per fluid particle initial value')
parser.add_argument('--k_B', default=1.0, type=float, help='Boltzmann constant initial value')  
args = parser.parse_args()     

device = torch.device('cpu')
# Net Parameters
dims = 3
model = CG_model(args, dims).to(device).float()
pt_file = argv[1]
checkpoint = torch.load(pt_file, map_location=device, weights_only=False)

#Saves model that calculates the forces and dS and Energy
model_jit_all = CG_model_jit_rand(args, dims).to(device).float()
model_jit_all.load_state_dict(checkpoint)
model = torch.jit.script(model_jit_all)
input_all = {'v': torch.rand((1000,3), device=device), 
            'S': torch.rand((1000,1), device=device), 
            'edge_index': torch.randint(0,3,(2,10000), device=device),
            'r_ij': torch.rand((10000,3), device=device),
            'd': torch.rand((1000,1), device=device)}
model.save('jit_models/sdpd_jit_rand.pt')

#Saves model that defines the volume
model_jit_w = CG_model_jit_w(args, dims).to(device).float()
model_jit_w.load_state_dict(checkpoint)
model = torch.jit.script(model_jit_w)
input_w = {'r_ij': torch.rand((1000,3), device=device),
           'edge_index': torch.randint(0,100,(2,1000), device=device, dtype=torch.int64),
           'N': torch.ones((1), device=device, dtype=torch.int64)* 100,}
model.save('jit_models/sdpd_jit_w.pt')

#Saves model that defines the entropy for the first step
model_jit_s = CG_model_jit_s(args, dims).to(device).float()
model_jit_s.load_state_dict(checkpoint)
model = torch.jit.script(model_jit_s)
input_s = {'r_ij': torch.rand((1000,3), device=device),
           'v': torch.rand((100,3), device=device),
           'edge_index': torch.randint(0,100,(2,1000), device=device, dtype=torch.int64)}
model.save('jit_models/sdpd_jit_s.pt')


