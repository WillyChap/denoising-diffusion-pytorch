import math
import copy
from pathlib import Path
from random import random
from functools import partial
from functools import lru_cache
import time 
import os
import subprocess

from collections import namedtuple
from multiprocessing import cpu_count
import glob
import pickle
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
import wandb

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__
from multiprocessing import cpu_count

# constants

# helpers functions

# gaussian diffusion trainer class
def get_num_cpus():
    # Check if PBS_NODEFILE exists
    if 'PBS_NODEFILE' in os.environ:
        nodefile = os.getenv('PBS_NODEFILE')
        try:
            # Use subprocess to count the lines (which correspond to CPUs)
            num_cpus = int(subprocess.check_output(['wc', '-l', nodefile]).split()[0])
            return num_cpus
        except Exception as e:
            print(f"Error reading PBS_NODEFILE: {e}")
            return None
    else:
        # Fallback to cpu_count if PBS_NODEFILE is not available
        from multiprocessing import cpu_count
        return cpu_count()

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules