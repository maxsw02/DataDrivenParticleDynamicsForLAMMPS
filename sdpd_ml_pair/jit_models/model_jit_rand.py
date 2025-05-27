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

        v = inputs['v']
        edge_index = inputs['edge_index']
        r_ij = inputs['r_ij']
        S = inputs['S']
        d = inputs['d']
        dW = inputs['dW']
        dV_ij = inputs['dV']


        N = v.size(0)
        edges = edge_index.size(1)

        i = edge_index[0,:]
        j = edge_index[1,:]
        k_B = torch.exp(self.log_k_B)
        m = torch.exp(self.log_m)

        v_ij = v[i] - v[j]
        EPS = 1e-3
        
        r_norm = self._dot(r_ij, r_ij)**0.5
        # Density: Kernel Approximation
        r_n_plus = r_norm + EPS
        r_n_minus = r_norm - EPS
        
        r_perturb = torch.cat([r_n_plus,r_n_minus], dim=0)  # [6*num_edges, 3]

        W_ij_perturb = self.model_W(r_perturb)
        W_plus = W_ij_perturb[0:edges]       # [N, 1]
        W_minus = W_ij_perturb[edges:2*edges]     # [N, 1]

        dW_dr = (W_plus - W_minus) / (2 * EPS * r_norm)  # [N, 1]
        grad_W_ij = dW_dr * r_ij
        grad_W_ji = grad_W_ij

        #if first_snap else inputs['S']

        e_ij = (r_ij / (r_norm + 1e-8))

        # Entropy estimation
        V = 1/d
        #print(V.size())

        EPS = 1e-2
        S_perturb = torch.cat([S,S+EPS,S,S-EPS], dim=0)  # [6*num_edges, 1]
        V_perturb = torch.cat([V,V,V+EPS,V], dim=0)  # [6*num_edges, 1]
        U_cat = self.model_E([S_perturb, V_perturb])
        U = U_cat[0:N]
        U_Splus = U_cat[N:2*N]
        U_Vplus = U_cat[2*N:3*N]
        U_Sminus = U_cat[3*N:4*N]

        T = (U_Splus - U) / EPS
        P = -(U_Vplus - U) / EPS
        C = T * EPS**2 / (U_Splus - 2*U + U_Sminus)

        termPd = P[i] / d[i]**2 * grad_W_ij + P[j] / d[j]**2 * grad_W_ji
        
        gradP_d = scatter_add(termPd, i, dim=0, dim_size=N) - scatter_add(termPd, j, dim=0, dim_size=N)

        T_i = T[i]
        T_j = T[j]
        EPS = 1e-3
        T_cat = torch.cat([T_i, T_j, T_i + EPS, T_j+ EPS ], dim=0)  # [2N, 1]
        r_cat = (r_norm/self.h).repeat(4, 1)

        A_cat = self.model_A([r_cat, T_cat])
        B_cat = self.model_B([r_cat, T_cat])
        C_cat = self.model_C([r_cat, T_cat])
        
        A_i       = A_cat[0:edges]
        A_j       = A_cat[edges:2*edges]
        A_i_eps   = A_cat[2*edges:3*edges]
        A_j_eps   = A_cat[3*edges:4*edges]

        B_i       = B_cat[0:edges]
        B_j       = B_cat[edges:2*edges]
        B_i_eps   = B_cat[2*edges:3*edges]
        B_j_eps   = B_cat[3*edges:4*edges]

        C_i       = C_cat[0:edges]
        C_j       = C_cat[edges:2*edges]
        C_i_eps   = C_cat[2*edges:3*edges]
        C_j_eps   = C_cat[3*edges:4*edges]

        A_ij = A_i * A_j
        B_ij = B_i * B_j
        C_ij = C_i * C_j
        gradA_T_i = 2. * A_ij * (A_i_eps * A_j - A_ij) / EPS
        gradB_T_i = 2. * B_ij * (B_i_eps * B_j - B_ij) / EPS
        gradC_T_i = 2. * C_ij * (C_i_eps * C_j - C_ij) / EPS

        gradA_T_j = 2. * A_ij * (A_i * A_j_eps - A_ij) / EPS
        gradB_T_j = 2. * B_ij * (B_i * B_j_eps - B_ij) / EPS
        gradC_T_j = 2. * C_ij * (C_i * C_j_eps - C_ij) / EPS

        auxMSV = 0.5 * A_ij**2  * v_ij + (0.5 *A_ij**2 + (B_ij**2 - A_ij**2) / self.D) * self._dot(e_ij, v_ij) * e_ij
        termMSV = (1./T_i + 1./T_j) * auxMSV

        MgradS_v = -0.5 * (scatter_add(termMSV, i,dim=0, dim_size=N) - scatter_add(termMSV, j,dim=0, dim_size=N))  
        #MgradS_v = -0.5 * (scatter_add(termMSV, i, dim_size=N) - scatter_add(termMSV, j, dim_size=N))  
        
        term = -(1/C[i]/T_i + 1/C[j]/T_j) * auxMSV
        term1 = scatter_add(term, i, dim=0,dim_size=N) - scatter_add(term, j, dim=0,dim_size=N)
        
        term2 = scatter_add((gradA_T_i/2 * v_ij + (gradA_T_i/2 + (gradB_T_i - gradA_T_i) / self.D) * self._dot(e_ij, v_ij) * e_ij) / C[i], i, dim=0,dim_size=N) \
            - scatter_add((gradA_T_j/2 * v_ij + (gradA_T_j/2 + (gradB_T_j - gradA_T_j) / self.D) * self._dot(e_ij, v_ij) * e_ij) / C[j], j,dim=0, dim_size=N)
        term3 = scatter_add((gradA_T_j/2 * v_ij + (gradA_T_j/2 + (gradB_T_j - gradA_T_j) / self.D) * self._dot(e_ij, v_ij) * e_ij) / C[j], i,dim=0, dim_size=N) \
                - scatter_add((gradA_T_i/2 * v_ij + (gradA_T_i/2 + (gradB_T_i - gradA_T_i) / self.D) * self._dot(e_ij, v_ij) * e_ij) / C[i], j, dim=0,dim_size=N)
        divM_v = -1/2 * (term1 + term2 + term3)
        #divM_v = 0.0001 * torch.rand((v.size(0),3), device=v.device)

        auxMSS = (A_ij**2/2 * self._dot(v_ij, v_ij) + (A_ij**2/2 + (B_ij**2 - A_ij**2) / self.D) * self._dot(e_ij, v_ij)**2) / 4
        MgradS_S = scatter_add((1/T[i] + 1/T[j]) * auxMSS + (1/T[i] - 1/T[j]) * C_ij**2, i, dim=0,dim_size=N) \
                  + scatter_add((1/T[j] + 1/T[i]) * auxMSS + (1/T[j] - 1/T[i]) * C_ij**2, j,dim=0, dim_size=N)
        
        #t_all = -(2/C[i]/T_i + 1/C[j]/T_j) * auxMSS \
        #+ gradA_T_i/2 * self._dot(v_ij, v_ij) + (gradA_T_i/2 + (gradB_T_i - gradA_T_i) / self.D) * self._dot(e_ij, v_ij)**2) / C[i] /4
        
        term1 = scatter_add(-(2/C[i]/T_i + 1/C[j]/T_j) * auxMSS, i, dim=0,dim_size=N) \
            + scatter_add(-(2/C[j]/T_j + 1/C[i]/T_i) * auxMSS, j, dim=0,dim_size=N)
        term2 = scatter_add((gradA_T_i/2 * self._dot(v_ij, v_ij) + (gradA_T_i/2 + (gradB_T_i - gradA_T_i) / self.D) * self._dot(e_ij, v_ij)**2) / C[i] /4, i,dim=0, dim_size=N) \
            + scatter_add((gradA_T_j/2 * self._dot(v_ij, v_ij) + (gradA_T_j/2 + (gradB_T_j - gradA_T_j) / self.D) * self._dot(e_ij, v_ij)**2) / C[j] /4, j,dim=0, dim_size=N)
        term3 = scatter_add((gradA_T_j/2 * self._dot(v_ij, v_ij) + (gradA_T_j/2 + (gradB_T_j - gradA_T_j) / self.D) * self._dot(e_ij, v_ij)**2) / C[j] /4, i,dim=0, dim_size=N) \
                + scatter_add((gradA_T_i/2 * self._dot(v_ij, v_ij) + (gradA_T_j/2 + (gradB_T_i - gradA_T_i) / self.D) * self._dot(e_ij, v_ij)**2) / C[i] /4, j, dim=0,dim_size=N)
        term4 = scatter_add(-(2/C[i]/T_i - 1/C[j]/T_j) * C_ij**2, i, dim=0,dim_size=N) \
            + scatter_add(-(2/C[j]/T_j - 1/C[i]/T_i) * C_ij**2, j,dim=0, dim_size=N)
        term5 = scatter_add(gradC_T_i / C[i] - gradC_T_j / C[j], i,dim=0, dim_size=N) \
            + scatter_add(gradC_T_j / C[j] - gradC_T_i / C[i], j, dim=0,dim_size=N)
        term = -((self.D + 1) * A_ij**2/2 + (B_ij**2 - A_ij**2) / self.D)

        term6 = scatter_add(term, i, dim=0,dim_size=N) \
        + scatter_add(term, j, dim=0,dim_size=N)
        divM_S = term1 + term2 + term3 + term4 + term5 + term6 / m
        
        dvdt = (-gradP_d + MgradS_v + k_B * divM_v) / m
        dSdt = (MgradS_S + k_B * divM_S) / T

        
        dW_ij_bar, tr_dW_ij = self.get_Wiener(dW,torch.tensor(edges))

        term = torch.einsum('...ij,...j->...i', A_ij[...,None] * dW_ij_bar + B_ij[...,None] * tr_dW_ij / self.D, e_ij)
        dv_tilde = (2 * k_B)**0.5 / m * (scatter_add(term, i, dim=0, dim_size=N) - scatter_add(term, j, dim=0, dim_size=N)) # dW_ij = dW_ji

        term = -1/2 * dot(term, v_ij)
        term1 = scatter_add(term, i, dim=0, dim_size=N) + scatter_add(term, j, dim=0, dim_size=N)
        term = C_ij * dV_ij

        term2 = scatter_add(term, i, dim=0, dim_size=N) - scatter_add(term, j, dim=0, dim_size=N) # dV_ij = -dV_ji  
        dS_tilde = (2 * k_B)**0.5 / T * (term1 + term2)

        E = (U + 1/2 * m * self._dot(v,v))
        sqrt_invdt = 1./torch.sqrt(self.dt)

        #return {'dvdt': dvdt , 'dSdt': dSdt, 'E': E, 'divM_v': divM_v, 'MgradS_v': MgradS_v, 'gradP_d': gradP_d, 'P': P, 'gradW_ij': grad_W_ij, 'gradW_ji': grad_W_ji, 'U': U, 'S ': S}
        #return {'dvdt': dvdt , 'dSdt': dSdt, 'E': E, 'S': S, 'd': d, 'dS_tilde': dS_tilde}
        #return {'dvdt': dvdt + dv_tilde, 'dSdt': dSdt + dS_tilde, 'E': E, 'divM_v': divM_v, 'MgradS_v': MgradS_v, 'gradP_d': gradP_d, 'P': P, 'gradW_ij': grad_W_ij, 'gradW_ji': grad_W_ji, 'U': U, 'S ': S}
        #return {'dvdt': dvdt , 'dSdt': dSdt, 'E': E}
        #return {'dvdt': dvdt, 'dSdt': dSdt, 'E': E,  'Aij': A_ij, 'Bij': B_ij, 'Cij': C_ij, 'k_b': k_B, 'm': m, 'T': T}
        return {'dvdt': dvdt + dv_tilde*sqrt_invdt, 'dSdt': dSdt + dS_tilde*sqrt_invdt, 'E': E}

    def _dot(self, a, b):
        '''
        Inner product of vectors a and b
        '''
        out = torch.einsum('...i,...i->...', a, b)[..., None]
        return out
    
    def get_Wiener(self, dW_ij,N):

        device = dW_ij.device

        I = torch.eye(self.D, device=device).repeat(N, 1, 1)
        tr_dW_ij = I * torch.einsum('...ii', dW_ij)[...,None, None]
        dW_ij_bar = 1/2 * (dW_ij + dW_ij.transpose(1,2)) - tr_dW_ij / self.D
        # Independent term
        #dV_ij = torch.ones([N, 1], device=device) * 0.1
        #dV_ij = torch.randn([N, 1], device=device)
    
        return dW_ij_bar, tr_dW_ij



if __name__ == '__main__':
    pass
