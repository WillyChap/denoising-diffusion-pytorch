from denoising_diffusion_pytorch.C_denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer_CESM
import torch
import requests
import random



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

def main():
    # Load Configuration
    config = get_config()
    print('...starting up...')
    print('... Think about writing a routine that shuffles files at the beginning of a run ...')
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
        do_wandb = True,
    )
    
    
    trainer.train()

if __name__ == "__main__":
    main()
