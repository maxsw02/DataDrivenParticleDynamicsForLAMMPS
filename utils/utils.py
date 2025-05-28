"""utils.py"""

import os
import argparse
import torch
import numpy as np


def str2bool(v):
    # Code from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def none_or_float(value):
    if value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: '{value}'")


def save_log(log, output_dir, name):

    file_name = name + '_log.txt'
    save_dir = os.path.join(output_dir, file_name)

    with open(save_dir, "w") as f:
        # Header keys
        f.write(' '.join(log.keys()) + '\n')
        
        # Values
        for values in zip(*log.values()):
            f.write(' '.join(f'{value}' for value in values) + '\n')



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


