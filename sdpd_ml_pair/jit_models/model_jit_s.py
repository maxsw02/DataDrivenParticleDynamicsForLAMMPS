"""model_jit.py"""

import torch
from torch import Tensor
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import softmax
import torch.nn.functional as F
from utils import dot, outer, eye_like
from typing import Optional, List, Dict
import torch.func as func


# Multi Layer Perceptron
class MLP(nn.Module):
    def __init__(self, layer_vec):
        super().__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k+1])
            self.layers.append(layer)
            if k != len(layer_vec) - 2: self.layers.append(nn.SiLU())

    def forward(self, x: List[Tensor]):
        x = torch.cat(x, dim=-1) if len(x) > 1 else x[0]
        for layer in self.layers:
            x = layer(x)
        return x


# Kernel Network
class KernelNet(nn.Module):
    def __init__(self, dim_hidden, n_hidden, h):
        super().__init__()
        
        self.MLP = MLP([1] + n_hidden * [dim_hidden] + [1])
        self.h = h


    def forward(self, r_ij):

        r = dot(r_ij, r_ij)**0.5
        S = r / self.h
        W = torch.exp(self.MLP([S])) * (1 - S**2)

        return W


# Gradient of a with respect to b
def grad(a: Tensor, b: Tensor) -> Tensor:
    '''
    Gradient of a with respect to b
    '''
    grad_outputs : List[Optional[Tensor]] = [torch.ones_like(a)]
    out = torch.autograd.grad([a,], [b], grad_outputs, retain_graph=True, create_graph=True)

    # Assert needed for JIT
    out = torch.zeros_like(b) if out is None else out[0]
    assert out is not None
    
    return out


# Equation of State Monotonic + Convex MLP
class MonotonicMLP(nn.Module):
    def __init__(self, layer_vec, t = [1, -1]):
        super().__init__()

        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k+1])
            self.layers.append(layer)

        self.register_buffer('t', torch.tensor(t))

    def forward(self, x: List[Tensor]):

        x = torch.cat(x, dim=-1) if len(x) > 1 else x[0]

        for i, layer in enumerate(self.layers):

            # Monotonicity: Modify weight matrix according to t
            if i == 0:
                weight = torch.abs(layer.weight) * self.t
            else:
                # Rest of weights should be positive
                weight = torch.abs(layer.weight)

            # Convexity: Convex activation function
            if i < len(self.layers) - 1: 
                x = F.softplus(F.linear(x, weight, layer.bias))
            else:
                # Last layer linear
                x = F.linear(x, weight, layer.bias)
        return x




# Machine Learning of Smoothed Dissipative Dynamics
class CG_model_jit(torch.nn.Module):
    def __init__(self, args, dims):
        super().__init__()

        self.D = dims        
        self.h = args.h
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.dt = args.dt
        self.boxsize = args.boxsize

        # Density
        self.model_W = KernelNet(self.dim_hidden, self.n_hidden, self.h)

        # Entropy teacher
        self.teacher = MLP([1 + self.D] + self.n_hidden * [self.dim_hidden] + [1])

        # Internal Energy
        self.model_E = MonotonicMLP([2] + self.n_hidden * [self.dim_hidden] + [1])
        
        # Stochastic Coefficients
        self.model_A = MLP([2] + self.n_hidden * [self.dim_hidden] + [1])
        self.model_B = MLP([2] + self.n_hidden * [self.dim_hidden] + [1])
        self.model_C = MLP([2] + self.n_hidden * [self.dim_hidden] + [1])

        # Extra constants
        self.log_k_B = nn.Parameter(torch.log(torch.tensor(args.k_B)))
        self.log_m = nn.Parameter(torch.log(torch.tensor(args.m)))


    def forward(self, inputs: Dict[str, Tensor]):

        edge_index = inputs['edge_index']
        r_ij = inputs['r_ij']
        v = inputs['v']
        v_ij = v[edge_index[0]] - v[edge_index[1]]
        N = v.size(0)
        #r_ = dot(r_ij, r_ij)**0.5 / self.h
        print(self.h)
        #print(v_ij)
        return self.model_S(r_ij, v_ij, edge_index, N)
        #return {'dvdt': dvdt , 'dSdt': dSdt, 'E': E, 'S': S, 'd': d, 'dS_tilde': dS_tilde}
        #return {'dvdt': dvdt + dv_tilde, 'dSdt': dSdt + dS_tilde, 'E': E}


    # Entropy encoder
    def model_S(self, r_ij, v_ij, edge_index, N:int):

        i = edge_index[0]
        j = edge_index[1]
        print(self.h)

        r_ = dot(r_ij, r_ij)**0.5 / self.h

        S = scatter_mean(self.teacher([r_, v_ij]), i, dim=0, dim_size=N) + \
            scatter_mean(self.teacher([r_, -v_ij]), j, dim=0, dim_size=N)

        return S
    

    # Noise Gaussian increments   
    def get_Wiener(self, edge_index):

        device = edge_index.device
        N = edge_index.size(-1)

        # Matrix of independent Wiener processes
        dW_ij = torch.randn([N, self.D, self.D], device=device)
        # Trace
        I = torch.eye(self.D, device=device).repeat(N, 1, 1)
        tr_dW_ij = I * torch.einsum('...ii', dW_ij)[...,None, None]
        #tr_dW_ij = I * dW_ij.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
        # Traceless symmetric part
        dW_ij_bar = 1/2 * (dW_ij + dW_ij.transpose(1,2)) - tr_dW_ij / self.D
        # Independent term
        dV_ij = torch.randn([N, 1], device=device)
    
        return dW_ij_bar, tr_dW_ij, dV_ij



if __name__ == '__main__':
    pass
