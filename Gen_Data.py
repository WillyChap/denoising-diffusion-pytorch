from denoising_diffusion_pytorch.C_denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer_CESM, num_to_groups
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


def get_config():
    config = {
        "input_channels": 1,
        "output_channels": 1,
        "context_image": True,
        "context_channels": 1,
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
        "batch_size" : 44,
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

def main():
    parser = argparse.ArgumentParser(description='Run CESM Diffusion Model')
    parser.add_argument('--month', type=int, default=1, help='Month to run the diffusion model on (1-12)')
    args = parser.parse_args()

    month_do = args.month
    
    config = get_config()
    print('...starting up...')
    print(f"Device Info: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

    model = Unet(
        channels = 3,
        dim = config['hidden_channels'],
        conditional_dimensions=1,
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
        folder,
        config,
        run_name,
        train_batch_size = config["batch_size"],
        train_lr = 5e-5,
        train_num_steps = 2,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False,           # whether to calculate fid during training
        max_grad_norm = 1.0,
        save_and_sample_every = 10,
        do_wandb = False,
    )

    do_run = 'yield_advantages'
    trainer.load('32','yield_advantages')

    # User setting: total number of samples to generate
    target_total_samples = 1000
    accumulated_samples = 0
    all_samples_list = []  # List to hold all samples

    Monther = xr.open_dataset('/glade/derecho/scratch/wchapman/CESM_LE2_vars/CESM_LE2_climo/CESM_LE2_climo_all_months.nc')
    replace_mats = (Monther.sel(time=month_do)['TREFHT'].values- 211.45393372)/(313.99099731-211.45393372)

    while accumulated_samples < target_total_samples:
        data, x_cond_rand = next(trainer.dl)

        #set x_cond to desired month. 

        x_cond_rand[:, :, :, :] = torch.tensor(replace_mats).to(trainer.device).unsqueeze(0).expand_as(x_cond_rand)

        # x_cond_rand = x_cond.to(trainer.device)

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

        saved_successfully = save_file(f'/glade/derecho/scratch/wchapman/Gen_CESM/samples_{do_run}_month{month_do:02}.nc', DS)
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



