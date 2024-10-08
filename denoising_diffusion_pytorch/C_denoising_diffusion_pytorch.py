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

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

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

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        PeriodicConv2d(dim, default(dim_out, dim), kernel_size=3, padding=1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class PeriodicBoundaryConv(nn.Module):
    def __init__(self, input_channels, init_dim, kernel_size=7, conditional_dimensions=0):
        super(PeriodicBoundaryConv, self).__init__()
        self.kernel_size = kernel_size
        # Compute padding based on the kernel size (integer division for "same" padding)
        self.padding = kernel_size // 2
        
        # Initialize Conv2d layer with the variable kernel size
        self.init_conv = nn.Conv2d(input_channels + conditional_dimensions, init_dim, kernel_size, padding=0)  # Set padding=0 as we handle it manually

    def forward(self, x):
        # x shape is [batch, variable, latitude, longitude]

        # 1. Circular padding for longitude (last dimension)
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='circular')  # Pad last dimension (longitude) by self.padding
        
        # 2. Zero padding (or reflect padding) for latitude (second-to-last dimension)
        x = F.pad(x, (0, 0, self.padding, self.padding), mode='reflect')  # Pad second-last dimension (latitude) by self.padding
        
        # 3. Apply convolution
        x = self.init_conv(x)
        
        return x

class PeriodicConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding=1):
        super(PeriodicConv2d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, padding=0)  # Padding is handled manually

    def forward(self, x):
        # Apply circular padding (periodic boundary condition) on longitude
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='circular')  # Apply padding to the last (longitude) dimension
        
        # Reflect padding or zero padding on latitude (2nd-to-last dimension)
        x = F.pad(x, (0, 0, self.padding, self.padding), mode='reflect')  # Reflect padding for latitude

        # Apply convolution
        return self.conv(x)


# model

class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        conditional_dimensions=0,
        self_condition = False,
        condition = True,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        self.condition = condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)

        # Old code (deprecated usage of padding):
        #self.init_conv = nn.Conv2d(input_channels + conditional_dimensions, init_dim, 7, padding = 3) #deprecated.
        
        self.init_conv = PeriodicBoundaryConv(input_channels, init_dim, kernel_size = 7, conditional_dimensions=conditional_dimensions)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else PeriodicConv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else PeriodicConv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None, x_cond=None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        if self.condition:
            x = torch.cat((x, x_cond),dim = 1)

        x = self.init_conv(x)
        
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)




def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        immiscible = False
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.condition = self.model.condition
        

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # immiscible diffusion

        self.immiscible = immiscible

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, x_cond=None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond, x_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, x_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond, x_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None, x_cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, x_cond = x_cond,clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, x_cond,return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            
            img, x_start = self.p_sample(img, t, self_cond, x_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, x_cond, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, x_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, x_cond, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), x_cond,return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            x_cond = x_cond if self.condition else None
            img, x_start = self.p_sample(img, i, self_cond, x_cond)

        return img

    def noise_assignment(self, x_start, noise):
        x_start, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, x_cond, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        ### maybe do the same for the condition: WEC KLUDGE

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond, x_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, x_cond, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, x_cond,*args, **kwargs)

# dataset classes

class DataProcessed(Dataset):
    def __init__(self, file_paths, config, mean_std_dict, min_max_dict):
        """
        Args:
            file_paths: List of paths to netCDF files
            config: Configuration dict with scaling and augmentation options
        """
        self.file_paths = file_paths
        self.config = config
        self.loaded_mean_std_dict = mean_std_dict
        self.loaded_min_max_dict = min_max_dict
        self.augmentation = config.get('augment', False)
        self.scaler = None
        if config['scaling'] == 'quantile':
            self.scaler = QuantileTransformer()

        # Initialize cache size (you can adjust it depending on memory constraints)
        self.cache_size = config.get('cache_size', 10)
        self.cached_data = {}  # Manual cache for storing loaded data

    def __len__(self):
        # Calculate total number of samples across all files
        total_len = 0
        for path in self.file_paths:
            with xr.open_dataset(path) as ds:
                total_len += ds.sizes['time']  # Assuming 'time' is the main dimension
        return total_len

    def _apply_scaling(self, ds):
        if self.config['scaling'] == 'std':
            ds['PS'][:,:,:] = (ds['PS'][:,:,:] - self.loaded_mean_std_dict['PS_mean']) / self.loaded_mean_std_dict['PS_std']
            ds['PRECT'][:,:,:] = (ds['PRECT'][:,:,:] - self.loaded_mean_std_dict['PRECT_mean']) / self.loaded_mean_std_dict['PRECT_std']
            ds['TREFHT'][:,:,:] = (ds['TREFHT'][:,:,:] - self.loaded_mean_std_dict['TREFHT_mean']) / self.loaded_mean_std_dict['TREFHT_std']
    
            vars = ['PS','PRECT','TREFHT']
    
            for vv in vars:
                mini = self.loaded_min_max_dict[f'{vv}_min']
                maxi = self.loaded_min_max_dict[f'{vv}_max']
                if maxi > 10:
                    maxi = 10
                if mini <-10:
                    mini = 10
                
                ds[vv][:,:,:] =  (ds[vv][:,:,:] - mini) / (maxi - mini)

            ds['month_expanded']=ds['month_expanded'].copy()
            ds['co2vmr_expanded']=ds['co2vmr_expanded'].copy()

            ds['month_expanded_scaled'] = (ds['month_expanded'][:,:,:] - 1)/(12-1)
            ds['co2vmr_expanded_scaled'] = (ds['co2vmr_expanded'][:,:,:] - 0.00039895)/(0.0008223-0.00039895)
            
        else:
            raise ValueError("Invalid scaling method specified.")
    
        return ds

    def _unapply_scaling(self, ds):
        if self.config['scaling'] == 'std':
            # Reverse the min-max scaling for 'PS', 'PRECT', and 'TREFHT'
            vars = ['PS', 'PRECT', 'TREFHT']
            
            for vv in vars:
                mini = self.loaded_min_max_dict[f'{vv}_min']
                maxi = self.loaded_min_max_dict[f'{vv}_max']
                
                if maxi > 10:
                    maxi = 10
                if mini < -10:
                    mini = -10
                
                # Reverse the min-max scaling
                ds[vv][:,:,:] = ds[vv][:,:,:] * (maxi - mini) + mini
    
            # Reverse the standardization (z-score normalization) for 'PS', 'PRECT', and 'TREFHT'
            ds['PS'][:,:,:] = ds['PS'][:,:,:] * self.loaded_mean_std_dict['PS_std'] + self.loaded_mean_std_dict['PS_mean']
            ds['PRECT'][:,:,:] = ds['PRECT'][:,:,:] * self.loaded_mean_std_dict['PRECT_std'] + self.loaded_mean_std_dict['PRECT_mean']
            ds['TREFHT'][:,:,:] = ds['TREFHT'][:,:,:] * self.loaded_mean_std_dict['TREFHT_std'] + self.loaded_mean_std_dict['TREFHT_mean']
    
        else:
            raise ValueError("Invalid scaling method specified.")
    
        return ds

    def _augment_data(self, data):
        # Apply rotations for augmentation
        data_rot90 = np.rot90(data, k=1, axes=(1, 2))
        data_rot180 = np.rot90(data, k=2, axes=(1, 2))
        data_rot270 = np.rot90(data, k=3, axes=(1, 2))
        
        # Concatenate all rotations along the first dimension (time)
        return np.concatenate([data, data_rot90, data_rot180, data_rot270], axis=0)

    @lru_cache(maxsize=2)  # Cache up to 5 file loads at once
    def _load_data_from_file(self, file_path):
        """
        Lazy load the data from file and preprocess it.
        """
        with xr.open_dataset(file_path) as ds:

            # Get the shape of the other variables (time, lat, lon)
            lat = ds['PS'].coords['lat']
            lon = ds['PS'].coords['lon']
            
            # Broadcast co2vmr across (time, lat, lon)
            co2vmr_broadcasted = ds['co2vmr'].broadcast_like(ds['PS'])
            
            # Alternatively, you can expand the dimensions manually
            co2vmr_expanded = ds['co2vmr'].expand_dims({'lat': lat, 'lon': lon}, axis=(1, 2))
            
            # Assign this expanded co2vmr back to the dataset with the correct dimensions
            ds['co2vmr_expanded'] = co2vmr_expanded
            
            ds['time.month'].broadcast_like(ds['PS'])
            # Alternatively, you can expand the dimensions manually
            month_expanded = ds['time.month'].expand_dims({'lat': lat, 'lon': lon}, axis=(1, 2))
            month_expanded = month_expanded-1
            month_corrected = month_expanded.where(month_expanded != 0, 12)
            ds['month_expanded'] = month_corrected
    
            ds = self._apply_scaling(ds)
            data = np.swapaxes(ds[['PS','PRECT','TREFHT']].to_array().values,0,1)  # Replace 'forecast' with the relevant key in your dataset
            cond = np.swapaxes(ds[['month_expanded_scaled','co2vmr_expanded_scaled']].to_array().values,0,1)  # Replace 'forecast' with the relevant key in your dataset
    
            # Apply augmentation if necessary
            if self.augmentation:
                data = self._augment_data(data)
    
        return torch.clamp(torch.tensor(data, dtype=torch.float32),min=0, max=1), torch.clamp(torch.tensor(cond, dtype=torch.float32),min=0, max=1)

    def __getitem__(self, idx):
        """
        Load data lazily and cache it, based on global index.
        """
        # Determine which file and sample this index belongs to
        cumulative_len = 0
        for file_idx, path in enumerate(self.file_paths):
            with xr.open_dataset(path) as ds:
                file_len = ds.sizes['time']  # Length along the 'time' dimension
                if cumulative_len + file_len > idx:
                    sample_idx = idx - cumulative_len
                    data, cond = self._load_data_from_file(path)  # Load data from cache or disk
                    return data[sample_idx], cond[sample_idx]
                cumulative_len += file_len

        raise IndexError(f"Index {idx} is out of bounds")

# trainer class
class Trainer_CESM:
    def __init__(
        self,
        diffusion_model,
        folder,
        config,
        run_name,
        do_wandb, 
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False
    ):
        super().__init__()

        # accelerator

        if do_wandb:
            self.accelerator = Accelerator(
                split_batches = split_batches,
                mixed_precision = mixed_precision_type if amp else 'no',
                log_with = "wandb"
            )
        else:
            self.accelerator = Accelerator(
                split_batches = split_batches,
                mixed_precision = mixed_precision_type if amp else 'no'
            )
        
        # model
        self.run_name = run_name
        self.do_wandb = do_wandb
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        
        self.config = config
        

        FPs = sorted(glob.glob(f'/{folder}/*.nc'))
        with open('./scaling/scaling_dict_CC.pkl', 'rb') as file:
            loaded_mean_std_dict = pickle.load(file)
        
        with open('./scaling/scaling_dict_minmax_CC.pkl', 'rb') as file:
            loaded_min_max_dict = pickle.load(file)
        
        self.ds = DataProcessed(FPs, config, loaded_mean_std_dict, loaded_min_max_dict)
        
        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        print('cpu count:', cpu_count())



        # Check if PBS_NP is set, otherwise fallback to cpu_count()
        # num_cpus = int(os.getenv('PBS_NP', cpu_count()))

        num_cpus = get_num_cpus()

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 64)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone, run_name):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'{run_name}-{milestone}.pt'))

    def load(self, milestone, run_name):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'{run_name}-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        self.config["num_params"]=sum(pttt.numel() for pttt in self.model.parameters() if pttt.requires_grad)
        # Initialize tracker via accelerator

        if self.do_wandb:
            self.accelerator.init_trackers(project_name="CESM_PS_PRECT_T2m", config=self.config,
                                           init_kwargs={"wandb":{"name":self.run_name}} )
    
            # Add a delay to allow the tracker to initialize properly
            print("Waiting for 10 seconds to ensure tracker is initialized...")
            time.sleep(10)

            # Fetch the tracker and check if the run is associated
            wandb_tracker = self.accelerator.get_tracker("wandb")
    
            # Ensure tracker has a run object
            if hasattr(wandb_tracker, 'run') and wandb_tracker.run:
                wandb_run = wandb_tracker.run
                run_name = wandb_run.name if wandb_run.name else "Unnamed run"
                print(f"Run Name: {run_name}")
            else:
                print("Error: The tracker does not have a Wandb run associated.")

        # Store the run name

        dada, x_c = self.ds.__getitem__(0)
        x_cond_rand = torch.zeros(60, self.config["context_channels"], x_c.shape[1], x_c.shape[2], device=device)
        num_gpus = accelerator.num_processes
        
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data, x_cond = next(self.dl)
                    data = data.to(device)
                    x_cond = x_cond.to(device)

                    # Perform a safe in-place update
                    go_in = torch.randint(0, 100, (1,)).item()  # Index for x_cond_rand
                    if go_in < 30:
                        idx_rand1 = torch.randint(0, 59, (1,)).item()  # Index for x_cond_rand
                        idx_rand2 = torch.randint(0, int(self.batch_size/num_gpus), (1,)).item()  # Index for x_cond
                        x_cond_rand[idx_rand1, :, :, :] = x_cond[idx_rand2, :, :, :].clone()  # Clone to avoid issues with inference mode

                    with self.accelerator.autocast():
                        loss = self.model(data, x_cond)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                if (self.step%10) == 0:
                    if self.do_wandb:
                        self.accelerator.log({'total_loss': total_loss}, step=self.step)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            x_cond_rand = x_cond_rand.to(device)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, x_cond = x_cond_rand[:n,:,:,:]), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        
                        samples_fig=plt.figure(figsize=(18, 9))
                        plt.suptitle("Samples after %d epochs" % self.step)
                        for pp in range(20):
                            plt.subplot(4, 5, 1 + pp)
                            plt.axis('off')
                            plt.imshow(all_images[pp,2,:,:].squeeze(0).data.cpu().numpy(),
                                    cmap='RdBu', vmin=0, vmax=1)
                            plt.colorbar()
                        plt.tight_layout()

                        if self.do_wandb:
                            self.accelerator.log({"Samples":wandb.Image(samples_fig)})

                        samples_fig=plt.figure(figsize=(18, 9))
                        plt.suptitle("Conditions after %d epochs" % self.step)
                        for pp in range(20):
                            plt.subplot(4, 5, 1 + pp)
                            plt.axis('off')
                            plt.imshow(x_cond_rand[pp,0,:,:].squeeze().data.cpu().numpy(),
                                    cmap='RdBu', vmin=0, vmax=1)
                            plt.colorbar()
                        plt.tight_layout()
                        if self.do_wandb:
                            self.accelerator.log({"Condition":wandb.Image(samples_fig)})
                        plt.close()

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')

                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone, self.run_name)

                pbar.update(1)

        accelerator.print('training complete')
