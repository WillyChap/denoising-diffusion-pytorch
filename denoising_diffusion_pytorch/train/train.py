import glob
import pickle
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler


from torchvision import utils

from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from denoising_diffusion_pytorch.utils import *
from denoising_diffusion_pytorch.datasets import DataProcessed
from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__

import os
import glob
import shutil
import random

def shuffle_and_copy(folder_path, shuffled_folder):
    # Get all .nc files in the folder, sorted
    file_paths = sorted(glob.glob(os.path.join(folder_path, '*.nc')))
    
    # Check if the shuffled folder exists
    if not os.path.exists(shuffled_folder):
        os.makedirs(shuffled_folder)

        # Shuffle the file paths
        shuffled_files = file_paths[:]
        random.shuffle(shuffled_files)

        # Copy the files with shuffled names
        for i, file_path in enumerate(shuffled_files):
            filename = os.path.basename(file_path)
            new_filename = f"{i:04d}_{filename}"  # Prepend the index to filename
            new_filepath = os.path.join(shuffled_folder, new_filename)
            shutil.copyfile(file_path, new_filepath)

        print(f"Copied and shuffled {len(file_paths)} files to {shuffled_folder}.")
    else:
        print(f"Shuffled folder '{shuffled_folder}' already exists. Re-shuffling files within the folder.")
        
        # If folder exists, get all the files there
        shuffled_file_paths = sorted(glob.glob(os.path.join(shuffled_folder, '*.nc')))
        
        # Shuffle again in-place
        random.shuffle(shuffled_file_paths)

        for i, file_path in enumerate(shuffled_file_paths):
            filename = os.path.basename(file_path)
            new_filename = f"{i:04d}_{filename.split('_', 1)[-1]}"  # Keep original filename, but update index
            new_filepath = os.path.join(shuffled_folder, new_filename)

            # If new filename differs, rename the file
            if file_path != new_filepath:
                os.rename(file_path, new_filepath)

        print(f"Re-shuffled files in {shuffled_folder}.")

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
        num_samples = 36,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        shuffle_files = False,
        gen_specific_samples = False,
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
        self.shuffle_files = shuffle_files
        self.run_name = run_name
        self.do_wandb = do_wandb
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.gen_specific_samples = gen_specific_samples
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
        self.folder = folder
        
        if self.accelerator.is_main_process:
            if shuffle_files:
                print('...shuffling files...')
                shuffle_and_copy(folder, f"{folder}_shuffled/")
                print('...sleeping...')
                time.sleep(5)
                print('...done sleeping...')
            else:
                FPs = sorted(glob.glob(f'/{folder}/*.nc'))

        self.accelerator.wait_for_everyone()
        if shuffle_files:
            FPS = sorted(glob.glob(f'/{folder}/_shuffled/*.nc'))
        else:
            FPS = sorted(glob.glob(f'/{folder}/*.nc'))
        
        with open('./scaling/scaling_dict_CC.pkl', 'rb') as file:
            loaded_mean_std_dict = pickle.load(file)
        
        with open('./scaling/scaling_dict_minmax_CC.pkl', 'rb') as file:
            loaded_min_max_dict = pickle.load(file)


        self.loaded_mean_std_dict = loaded_mean_std_dict
        self.loaded_min_max_dict = loaded_min_max_dict
        self.train_batch_size = train_batch_size
        self.ds = DataProcessed(FPS, config, loaded_mean_std_dict, loaded_min_max_dict)
        
        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        print('cpu count:', cpu_count())


        # Check if PBS_NP is set, otherwise fallback to cpu_count()
        # num_cpus = int(os.getenv('PBS_NP', cpu_count()))

        num_cpus = get_num_cpus()

        dl = DataLoader(self.ds, batch_size = train_batch_size,shuffle=False, pin_memory = True, num_workers = 64)
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

                        #from here ... regen and save ....

                        if self.gen_specific_samples:
    
                            x_cond_rand = x_cond_rand*0
    
                            mdo = 2
                            mdot = (mdo - 1)/(12-1)
    
                            co2_ = 0.0004020489956418128
                            co2_t = (co2_- 0.00039895)/(0.0008223-0.00039895)
                            
                            x_cond_rand[:,0,:,:] = mdot
                            x_cond_rand[:,1,:,:] = co2_t
                            
                            with torch.inference_mode():
                                
                                milestone = self.step // self.save_and_sample_every
                                batches = num_to_groups(self.num_samples, self.batch_size)
                                print(f'!!!!!! batches is: {batches} !!!!!!!!')
                                x_cond_rand = x_cond_rand.to(device)
                                all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, x_cond = x_cond_rand[:n,:,:,:]), batches))
    
                            all_images = torch.cat(all_images_list, dim = 0)
                            # Extract variables PS, PRECT, TREFHT
                            PS = all_images[:, 0, :, :].squeeze(0).data.cpu().numpy()
                            PRECT = all_images[:, 1, :, :].squeeze(0).data.cpu().numpy()
                            TREFHT = all_images[:, 2, :, :].squeeze(0).data.cpu().numpy()
                    
                            num_samples_, height_, width_ = PS.shape  # Assume the same shape for all variables
                    
                            # Use an example dataset to load latitudes and longitudes
                            print(f'folder looks like: {self.folder}')
                            samp_ds = xr.open_dataset(glob.glob(f'{self.folder}/b.e21.BSSP370cmip6.*.nc')[0])
                            latitudes = samp_ds['lat']
                            longitudes = samp_ds['lon']
                            samples_ = np.arange(num_samples_)
                    
                            # Create the xarray Dataset
                            ds_ = xr.Dataset(
                                {
                                    'PS': (('samples', 'lat', 'lon'), PS),
                                    'PRECT': (('samples', 'lat', 'lon'), PRECT),
                                    'TREFHT': (('samples', 'lat', 'lon'), TREFHT)
                                },
                                coords={
                                    'samples': samples_,
                                    'lat': latitudes,
                                    'lon': longitudes
                                }
                            )
                            DS = self.ds._unapply_scaling(ds_)
                            DS.to_netcdf(str(self.results_folder / f'{self.run_name}-{milestone}_month_{mdo:02}_co2_{co2_}.nc'))
                        else:
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
            
                self.accelerator.wait_for_everyone()
                pbar.update(1)
                

        accelerator.print('training complete')