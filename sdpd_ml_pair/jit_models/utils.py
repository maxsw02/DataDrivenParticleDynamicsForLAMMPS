"""utils.py"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np


def str2bool(v):
    # Code from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_log(log, output_dir, name):

    file_name = name + '_log.txt'
    save_dir = os.path.join(output_dir, file_name)

    with open(save_dir, "w") as f:
        # Header keys
        f.write(' '.join(log.keys()) + '\n')
        
        # Values
        for values in zip(*log.values()):
            f.write(' '.join(f'{value}' for value in values) + '\n')


# Plot simulations
def plot_sim(results_net, results_gt, save_dir, epoch):

    # Load network prediction
    x_net, x_gt = results_net['x'], results_gt['x']
    p_sum_net, E_sum_net, S_sum_net = results_net['p_sum'], results_net['E_sum'], results_net['S_sum']

    T = x_gt.size(0)
    #T = x_net.size(0)
    N = x_net.size(1)

    # Load ground truth
    flag = True if 'p_sum' in results_gt.keys() else False
    if flag: p_sum_gt, E_sum_gt, S_sum_gt = results_gt['p_sum'], results_gt['E_sum'], results_gt['S_sum']

    VACF_net, RDF_net, r_RDF_net, MSD_net = results_net['VACF'], results_net['RDF'][0], results_net['RDF'][1], results_net['MSD']
    VACF_gt, RDF_gt, r_RDF_gt, MSD_gt = results_gt['VACF'], results_gt['RDF'][0], results_gt['RDF'][1], results_gt['MSD']

    D = p_sum_net.size(-1)      

    nodes = [0, N//4, N//2, N-1]
    #nodes = [0]
    colors = ['r', 'g', 'b']

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for node in nodes:
        # Position
        for dim in range(D):
            axs[0,0].plot(x_net[:T, node, dim], colors[dim] + '.')
            axs[0,0].plot(x_gt[:T, node, dim], colors[dim])
        axs[0,0].set_title('Position')
        axs[0,0].set_xlabel('t') 
        axs[0,0].grid(True)

        # Velocity
        for dim in range(D):
            axs[0,1].plot(x_net[:T, node, D + dim], colors[dim] + '.')
            axs[0,1].plot(x_gt[:T, node, D + dim], colors[dim])
        axs[0,1].set_title('Velocity')
        axs[0,1].set_xlabel('t')  
        axs[0,1].grid(True)

        # Entropy
        axs[0,2].plot(x_net[:T, node, 2*D], 'r.')
        axs[0,2].plot(x_gt[:T, node, 2*D], 'r')
        if flag: axs[0,2].plot(x_gt[:, node, 2*D], 'r')
        axs[0,2].set_title('Entropy')
        axs[0,2].set_xlabel('t')  
        axs[0,2].grid(True)

    # Momentum Conservation
    for dim in range(D):
        p_net = p_sum_net - p_sum_net[0]
        if flag: p_gt = p_sum_gt - p_sum_gt[0]
        axs[1,0].plot(p_net[:T, dim], colors[dim] + '.')
        if flag: axs[1,0].plot(p_gt[:T, dim], colors[dim]) 
    axs[1,0].set_title('Sum Momentum')
    axs[1,0].set_xlabel('t')
    axs[1,0].grid(True)  

    # Energy Conservation
    E_net = (E_sum_net - E_sum_net[0]) * 100 / abs(E_sum_net[0] + 1e-8)
    if flag: E_gt = (E_sum_gt - E_sum_gt[0]) * 100 / abs(E_sum_gt[0] + 1e-8)
    axs[1,1].plot(E_net[:T], 'r.', label='Net')
    if flag: axs[1,1].plot(E_gt[:T], 'r', label='GT')
    axs[1,1].set_title('Sum Energy (%)')
    axs[1,1].set_xlabel('t')  
    axs[1,1].grid(True)

    # Entropy Conservation
    S_net = (S_sum_net - S_sum_net[0]) * 100 / abs(S_sum_net[0] + 1e-8)
    if flag: S_gt = (S_sum_gt - S_sum_gt[0]) * 100 / abs(S_sum_gt[0] + 1e-8)
    axs[1,2].plot(S_net[:T], 'r.', label='Net') 
    if flag: axs[1,2].plot(S_gt[:T], 'r', label='GT')
    axs[1,2].set_title('Sum Entropy (%)')
    axs[1,2].set_xlabel('t')  
    axs[1,2].grid(True)

    # Labels
    axs[0,0].plot([], 'r.', label='Net'), axs[0,0].plot([], 'r', label='GT'), axs[0,0].legend()
    axs[0,1].plot([], 'r.', label='Net'), axs[0,1].plot([], 'r', label='GT'), axs[0,1].legend()
    axs[0,2].plot([], 'r.', label='Net'), 
    if flag: axs[0,2].plot([], 'r', label='GT')
    axs[0,2].legend()
    axs[1,0].plot([], 'r.', label='Net') 
    if flag: axs[1,0].plot([], 'r', label='GT') 
    axs[1,0].legend()
    axs[1,1].legend()
    axs[1,2].legend()

    plt.tight_layout()
    # Save plot
    plt.savefig(os.path.join(save_dir, 'plot_{}.png'.format(epoch)))


    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # VACF
    axs[0].plot(VACF_net, 'r.', label='Net')
    axs[0].plot(VACF_gt, 'b', label='GT')
    axs[0].set_title('VACF')
    axs[0].set_xlabel('t')  
    axs[0].grid(True)

    # RDF
    axs[1].plot(r_RDF_net, RDF_net, 'r.', label='Net')
    axs[1].plot(r_RDF_gt, RDF_gt, 'b', label='GT')
    axs[1].set_title('RDF')
    axs[1].set_xlabel('r')  
    axs[1].grid(True)

    # MSD
    axs[2].plot(MSD_net, 'r.', label='Net')
    axs[2].plot(MSD_gt, 'b', label='GT')
    axs[2].set_title('MSD')
    axs[2].set_xlabel('t')  
    axs[2].grid(True)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.tight_layout()
    # Save plot
    plt.savefig(os.path.join(save_dir, 'plot_{}.png'.format(epoch)))


# Plot Equation of State
def plot_EoS(model, output_dir):

    S_limits = (-5, 5)
    V_limits = (1/500, 10)
    samples = 10

    device = next(model.model_A.parameters()).device
    S_range = torch.linspace(S_limits[0], S_limits[1], samples, device=device)
    V_range = torch.linspace(V_limits[0], V_limits[1], samples, device=device)
    S, V = torch.meshgrid(S_range, V_range, indexing='ij')
    S = S.flatten()[...,None]
    V = V.flatten()[...,None]
    S.requires_grad = True
    V.requires_grad = True

    E_net = model.model_E(S, V)
    P_net = -grad(E_net, V)
    T_net = grad(E_net, S)

    S = S.detach().cpu().reshape(samples, samples)
    V = V.detach().cpu().reshape(samples, samples)
    E_net = E_net.detach().cpu().reshape(samples, samples)
    P_net = P_net.detach().cpu().reshape(samples, samples)
    T_net = T_net.detach().cpu().reshape(samples, samples)

    fig = plt.figure(figsize=(15, 5))

    # Plot ground truth
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(S, V, E_net, alpha=0.5, label='Net')
    ax1.set_title('Internal Energy')
    ax1.set_xlabel('S'), ax1.set_ylabel('V'), ax1.set_zlabel('E(S, V)')

    # Plot ground truth
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(S, V, P_net, alpha=0.5, label='Net')
    ax2.set_title('Pressure')
    ax2.set_xlabel('S'), ax2.set_ylabel('V'), ax2.set_zlabel('P')

    # Plot predicted function
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(S, V, T_net, alpha=0.5, label='Net')
    ax3.set_title('Temperatue')
    ax3.set_xlabel('S'), ax3.set_ylabel('V'), ax3.set_zlabel('T')

    ax1.legend()
    ax2.legend()
    ax3.legend()

    # Save plot
    save_dir = os.path.join(output_dir, 'plot_EoS.png')
    plt.savefig(save_dir)


# Plot Kernel
def plot_kernel(model, output_dir):

    R_limits = (-model.h, model.h)
    samples = 50

    device = next(model.model_A.parameters()).device
    R = torch.linspace(R_limits[0], R_limits[1], samples, device=device)[...,None]

    W_net = model.model_W(R).detach().cpu()

    # Plot ground truth
    plt.figure(figsize=(5, 5))
    plt.plot(R.cpu(), W_net, label='Net')
    plt.title('Kernel')
    plt.xlabel('r'), plt.grid()
    plt.legend()

    # Save plot
    save_dir = os.path.join(output_dir, 'plot_kernel.png')
    plt.savefig(save_dir)


# Plot loss
def plot_loss(train_log, val_log, output_dir):
        
    train_mu = np.array(train_log['loss_mu'])
    val_mu = np.array(val_log['loss_mu'])

    train_var = np.array(train_log['loss_var'])
    val_var = np.array(val_log['loss_var'])

    train_mse = np.array(train_log['loss_mse'])
    val_mse = np.array(val_log['loss_mse'])

    # Plotting
    plt.figure(figsize=(15, 5))

    plt.subplot(1,3,1)
    plt.loglog(train_mu, label='Train')
    plt.loglog(val_mu, label='Val')
    plt.xlabel('Epoch')
    plt.title('Mean Loss')
    plt.legend(), plt.grid()

    plt.subplot(1,3,2)
    plt.semilogx(train_var, label='Train')
    plt.semilogx(val_var, label='Val')
    plt.xlabel('Epoch')
    plt.title('Var Loss')
    plt.legend(), plt.grid()

    plt.subplot(1,3,3)
    plt.loglog(train_mse, label='Train')
    plt.loglog(val_mse, label='Val')
    plt.xlabel('Epoch')
    plt.title('MSE Loss')
    plt.legend(), plt.grid()

    plt.tight_layout()

    # Save plot
    save_dir = os.path.join(output_dir, 'loss.png')
    plt.savefig(save_dir)


def dot(a, b):
    '''
    Inner product of vectors a and b
    '''
    # out = (a * b).sum(-1, keepdim=True)
    out = torch.einsum('...i,...i->...', a, b)[..., None]
    return out


def outer(a, b):
    '''
    Outer product of vectors a and b
    '''
    # out = a[...,None] @ torch.transpose(b[...,None],1,2)
    out = torch.einsum('...i,...j->...ij', a, b)
    return out


def grad(a, b):
    '''
    Gradient of a with respect to b
    '''
    out = torch.autograd.grad(a, b, torch.ones_like(a), retain_graph=True, create_graph=True)[0]
    return out


def eye_like(a):
    '''
    Identity matrix matching dimensions and devise with a
    '''
    I = torch.eye(a.size(1), device=a.device).repeat(a.size(0), 1, 1)
    return I


