from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion_CESM, Trainer_CESM
import torch
import matplotlib.pyplot as plt


def get_config():
    config = {
        "input_channels": 1,
        "output_channels": 1,
        "context_image": True,
        "context_channels": 1,
        "num_blocks": [2, 2],
        "hidden_channels": 32,
        "hidden_context_channels": 8,
        "time_embedding_dim": 256,
        "image_size": 128,
        "noise_sampling_coeff": 0.85,
        "denoise_time": 970,
        "activation": "gelu",
        "norm": True,
        "subsample": 100000,
        "save_name": "model_weights.pt",
        "dim_mults": [4, 4],
        "base_dim": 32,
        "timesteps": 1000,
        "pading": "reflect",
        "scaling": "std",
        "optimization": {
            "epochs": 400,
            "lr": 0.01,
            "wd": 0.05,
            "batch_size": 32,
            "scheduler": True
        }
    }
    return config

def main():
    # Load Configuration
    config = get_config()
    print('...starting up...')
    print(f"Device Info: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

    model = Unet(
        channels = 3,
        dim = 64,
        dim_mults = (2, 4, 6, 8),
        flash_attn = True
    )

    diffusion = GaussianDiffusion_CESM(
        model,
        image_size = (192, 288),
        timesteps = 1000,    # number of steps
        auto_normalize = True,
        objective = "pred_v",
     )
    diffusion.is_ddim_sampling = True
    print('model params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    trainer = Trainer_CESM(
        diffusion,
        '/glade/derecho/scratch/wchapman/CESM_LE2_vars/',
        config,
        train_batch_size = 32,
        train_lr = 5e-5,
        train_num_steps = 1,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False,           # whether to calculate fid during training
        max_grad_norm = 1.0,
    )

    trainer.train()

    trainer.load('1')
    sampled_images = diffusion.sample(batch_size = 4, return_all_timesteps = False)

    plt.pcolor(sampled_images[0,0,:,:].cpu())
    plt.colorbar()
    plt.savefig('./testfig.png')
    plt.close()

    plt.pcolor(sampled_images[0,1,:,:].cpu())
    plt.colorbar()
    plt.savefig('./testfig1.png')
    plt.close()

    plt.pcolor(sampled_images[0,2,:,:].cpu())
    plt.colorbar()
    plt.savefig('./testfig2.png')
    plt.close()
    

if __name__ == "__main__":
    main()
