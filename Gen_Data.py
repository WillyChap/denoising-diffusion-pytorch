from denoising_diffusion_pytorch.train import Trainer_CESM
from denoising_diffusion_pytorch.diffusion import GaussianDiffusion
from denoising_diffusion_pytorch.models import Unet

import torch
import requests
import random
from ema_pytorch import EMA
import torch.distributed as dist
from accelerate import Accelerator
import numpy as np
import xarray as xr
import glob
import argparse
import os
import sys
import multiprocessing
import signal
import threading
import torch.distributed as dist
import multiprocessing
import time
import sys
import argparse

##global settings.
do_run = 'governance_indexes'

repeate_co2 = False


def get_config():
    config = {
        "input_channels": 1,
        "output_channels": 1,
        "context_image": True,
        "context_channels": 2,
        "num_blocks": [2, 2],
        "hidden_channels": 64,
        "hidden_context_channels": 8,
        "time_embedding_dim": 256,
        "image_size": 128,
        "noise_sampling_coeff": 0.85,
        "denoise_time": 970,
        "activation": "gelu",
        "norm": True,
        "subsample": 100000,
        "save_name": "model_weights.pt",
        "dim_mults": (2, 4, 6, 8),
        "flash_attn": True,
        "base_dim": 32,
        "timesteps": 1000,
        "pading": "reflect",
        "scaling": "std",
        "batch_size" : 128,
        "dropout": 0.,
        "optimization": {
            "epochs": 400,
            "lr": 0.01,
            "wd": 0.05,
            "scheduler": True
        }
    }
    return config


def destroy_process_group():
    if dist.is_initialized():
        print("Destroying distributed process group...")
        dist.destroy_process_group()

def cleanup_and_exit():
    print("Cleaning up...")

    # Step 1: Destroy the distributed process group
    destroy_process_group()

    # Step 2: Kill any background threads (Avoid using thread._stop())
    for thread in threading.enumerate():
        if thread != threading.main_thread():
            print(f"Stopping thread: {thread.name}")
            # Instead of _stop(), try to gracefully terminate threads if possible

    # Step 3: Kill any child processes
    for process in multiprocessing.active_children():
        print(f"Terminating process with PID: {process.pid}")
        process.terminate()  # Graceful termination
        os.kill(process.pid, signal.SIGKILL)  # Forcefully kill the process

    # Step 4: Finally, kill the current process
    print("Forcefully terminating the main process.")
    os.kill(os.getpid(), signal.SIGKILL)

def generate_unique_filename(base_filename):
    """
    Generate a unique filename by appending an ensemble number (_001, _002, etc.)
    if the file already exists.
    """
    # Split the base filename into the base name and extension
    base, ext = os.path.splitext(base_filename)
    
    counter = 1
    unique_filename = f"{base}_{counter:03}{ext}"
    
    # Increment the counter until a unique filename is found
    while os.path.exists(unique_filename):
        counter += 1
        unique_filename = f"{base}_{counter:03}{ext}"
    
    return unique_filename

def save_file(filename, DS):
    # Generate a unique filename if the file already exists
    filename = generate_unique_filename(filename)
    
    try:
        DS.to_netcdf(filename)
        print(f"Final dataset saved as {filename}")
        time.sleep(25)
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

class FileAlreadyBuiltError(Exception):
    """Exception raised when the file has already been built."""
    pass

def main():
    
    parser = argparse.ArgumentParser(description='Run CESM Diffusion Model')
    parser.add_argument('--month', type=int, default=1, help='Month to run the diffusion model on (1-12)')
    parser.add_argument('--co2', type=float, default=0.000398996, help='co2vmr to run the diffusion model on (0.00039895 - 0.0008223)')
    parser.add_argument('--run_num', type=float, default=0.000398996, help='run number to test')
    args = parser.parse_args()
    run_num =  int(args.run_num)
    # Validate month
    if args.month < 1 or args.month > 12:
        print(f"Error: Invalid month {args.month}. Should be between 1 and 12.")
        sys.exit(1)
    
    # Validate co2
    if args.co2 < 0.00039895 or args.co2 > 0.0008223:
        print(f"Error: Invalid co2vmr {args.co2}. Should be between 0.00039895 and 0.0008223.")
        sys.exit(1)

    month_do = args.month
    co2 = args.co2
    
    fname = sorted(glob.glob(f'/glade/derecho/scratch/wchapman/Gen_CESM/samples_{do_run}_{run_num:03}_month{month_do:02}_co2{co2}*'))
    if not repeate_co2 and len(fname) >= 1:
        raise FileAlreadyBuiltError(f"File(s) already built for run {do_run}, month {month_do:02}, and co2 level {co2}. Exiting...")
    else:
        print('....start your engines....')

    
    
    config = get_config()
    print('...starting up...')
    print(f"Device Info: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

    model = Unet(
        channels = 3,
        dim = config['hidden_channels'],
        conditional_dimensions=config['context_channels'],
        dim_mults = config['dim_mults'],
        flash_attn = config['flash_attn'],
        dropout = config['dropout'],
        condition = True
    )
    
    diffusion = GaussianDiffusion(
        model,
        image_size = (192, 288),
        timesteps = config["timesteps"],    # number of steps
        auto_normalize = True,
        objective = "pred_v",
     )

    folder = '/glade/derecho/scratch/wchapman/CESM_LE2_vars_with_climo/'
    
    word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
    response = requests.get(word_site)
    WORDS = response.content.splitlines()
    
    # Generate random indices
    rn1 = random.randint(0, len(WORDS) - 1)
    rn2 = random.randint(0, len(WORDS) - 1)
    
    # Decode the byte lines to strings
    w1 = WORDS[rn1].decode('utf-8')
    w2 = WORDS[rn2].decode('utf-8')

    run_name = f'{w1}_{w2}'

    print(f'my name is {run_name}')
    
    diffusion.is_ddim_sampling = True
    print('model params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    trainer = Trainer_CESM(
        diffusion,
        '/glade/derecho/scratch/wchapman/CESM_LE2_vars_BSSP370cmip6/',
        config,
        run_name,
        train_batch_size = config["batch_size"],
        results_folder = './results_cc/',
        train_lr = 5e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False,           # whether to calculate fid during training
        max_grad_norm = 1.0,
        save_and_sample_every = 1000,
        do_wandb = False,
        gen_specific_samples = False,
    )

    
    trainer.load(run_num,do_run)

    # User setting: total number of samples to generate
    target_total_samples = 500
    accumulated_samples = 0
    all_samples_list = []  # List to hold all samples

    ds = xr.open_dataset(f'{trainer.folder}/b.e21.BSSP370cmip6.f09_g17.LE2-1231.004.cam.h0.CO2_PRECT_TREFHT_PS.201501-202412.nc')

   # Convert the scalar into a DataArray
    month_do_array = xr.DataArray(month_do)

    # Broadcast the scalar across the shape of PS (time, lat, lon)
    month_broadcasted = month_do_array.broadcast_like(ds['PS'])[0,:,:].values

    month_broadcasted = (month_broadcasted - 1)/(12-1)
    
    # Convert the scalar into a DataArray
    co2_do_array = xr.DataArray(co2)

    # Broadcast the scalar across the shape of PS (time, lat, lon)
    co2_broadcasted = co2_do_array.broadcast_like(ds['PS'])[0,:,:].values
    co2_broadcasted = (co2_broadcasted - 0.00039895)/(0.0008223-0.00039895)

    while accumulated_samples < target_total_samples:
        data, x_cond_rand = next(trainer.dl)
        batch_size = x_cond_rand.shape[0]  
        lat_dim = x_cond_rand.shape[2]     # This is 192
        lon_dim = x_cond_rand.shape[3]     # This is 288

        # month_broadcasted should be expanded to 4D shape: (batch, 1, lat, lon)
        month_broadcasted_tensor = torch.tensor(month_broadcasted).to(trainer.device)
        
        # Expand to match (batch_size, 1, lat, lon)
        month_broadcasted_tensor = month_broadcasted_tensor.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, lat_dim, lon_dim)
        
        # Now assign it to x_cond_rand[:, 0, :, :]
        x_cond_rand[:, 0, :, :] = month_broadcasted_tensor.squeeze(1)
        
        # Similarly, handle CO2 broadcasting
        co2_broadcasted_tensor = torch.tensor(co2_broadcasted).to(trainer.device)
        
        # Expand to match (batch_size, 1, lat, lon)
        co2_broadcasted_tensor = co2_broadcasted_tensor.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, lat_dim, lon_dim)
        
        # Assign it to x_cond_rand[:, 1, :, :]
        x_cond_rand[:, 1, :, :] = co2_broadcasted_tensor.squeeze(1)

        # Get local batch size for this process
        local_batch_size = trainer.batch_size // trainer.accelerator.num_processes

        # Sample on each process
        all_images_list = diffusion.sample(batch_size=local_batch_size, x_cond=x_cond_rand[:local_batch_size, :, :, :])

        # Gather all sampled data across processes
        gathered_samples = [torch.zeros_like(all_images_list) for _ in range(trainer.accelerator.num_processes)]
        dist.all_gather(gathered_samples, all_images_list)

        if trainer.accelerator.is_main_process:
            # Concatenate the results from all processes
            all_samples = torch.cat(gathered_samples, dim=0)

            # Accumulate samples until the target is reached
            accumulated_samples += all_samples.shape[0]
            all_samples_list.append(all_samples.cpu().numpy())  # Save the NumPy version of the samples

            print(f"Accumulated {accumulated_samples}/{target_total_samples} samples")

    if trainer.accelerator.is_main_process:
        # Concatenate all batches to create the final dataset
        final_samples = np.concatenate(all_samples_list, axis=0)

        # Truncate the array if more samples were generated than needed
        final_samples = final_samples[:target_total_samples]

        # Extract variables PS, PRECT, TREFHT
        PS = final_samples[:, 0, :, :]
        PRECT = final_samples[:, 1, :, :]
        TREFHT = final_samples[:, 2, :, :]

        num_samples, height, width = PS.shape  # Assume the same shape for all variables

        # Use an example dataset to load latitudes and longitudes
        samp_ds = xr.open_dataset(glob.glob(f'{folder}/b.e21.BHISTcmip6.*.nc')[0])
        latitudes = samp_ds['lat']
        longitudes = samp_ds['lon']
        samples = np.arange(num_samples)

        # Create the xarray Dataset
        ds = xr.Dataset(
            {
                'PS': (('samples', 'lat', 'lon'), PS),
                'PRECT': (('samples', 'lat', 'lon'), PRECT),
                'TREFHT': (('samples', 'lat', 'lon'), TREFHT)
            },
            coords={
                'samples': samples,
                'lat': latitudes,
                'lon': longitudes
            }
        )

        DS = trainer.ds._unapply_scaling(ds)
        # DS.to_netcdf(f'samples_{do_run}_month{month_do:02}.nc')
        # print(f"Final dataset saved as samples_{do_run}_month{month_do:02}.nc")

        saved_successfully = save_file(f'/glade/derecho/scratch/wchapman/Gen_CESM/samples_{do_run}_{run_num:03}_month{month_do:02}_co2{co2}.nc', DS)
    # After saving the file
    if saved_successfully:
        print("File saved, now quitting forcefully.")
        # trainer.accelerator.wait_for_everyone()
        cleanup_and_exit()
        # destroy_process_group()
        sys.exit(0)  #
    print("File saved, now quitting forcefully.")
    # trainer.accelerator.wait_for_everyone()
    cleanup_and_exit()
    # destroy_process_group()
    sys.exit(0)  #


if __name__ == "__main__":
    main()



