<img src="./images/cesm-diffusion.png" width="500px"></img>

## Conditional Denoising Diffusion Model for Climate Prediction, in Pytorch

This fork adapts the [Denoising Diffusion Probabilistic Model](https://arxiv.org/abs/2006.11239) for climate predictions using the CESM2 Large Ensemble (LENS) data. It predicts T2m, Precipitation, and Surface Pressure, conditioned on a given month and CO2 concentration (CO2vmr). This enables climate model forecasts with diffusion-based predictions, improving upon existing methodologies for ensemble predictions.

### Features
- **Conditional**: The model takes CO2vmr and month as inputs.
- **Training scripts for NCAR HPC**: New launch and training scripts tailored for the NCAR High Performance Computing environment.
- **File Structure**: The file tree has been significantly restructured to accommodate the new scripts and features.

For more information about the CESM2 Large Ensemble, visit the [CESM2 LENS page](https://www.cesm.ucar.edu/community-projects/lens2).

---

### Install

```bash
$ pip install .
```
## File Tree

```bash
.Gen_Data.py                # Generate samples once trained
.Train_CESM_Conditioned.py  # training control script
.launch.sh                  # PBS launch code
./notebooks/
├── Developement.ipynb      # development code.
├── Examining_Output.ipynb  # examining the output of model
├── Gather_CESM_Data.ipynb  # retrieve and stage variables, provide scaling pkl files 
./scaling/
├── scaling_file.pkl        # pkl files for scaling
./denoising_diffusion_pytorch/
├── datasets/               # Contains data loading and preprocessing scripts
├── diffusion/              # Core diffusion model code and sampling logic
├── models/                 # Defines the architecture (e.g., U-Net models)
├── train/                  # Training scripts, including multi-GPU support
├── utils/                  # Utility functions (logging, metrics, etc.)
└── .github/                # GitHub-specific files (actions, issues templates)
```


## Usage

I have provided, launch scripts for PBS scheduling, and wandb implementation, but the user must input their own details. 


```python
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
    rn1 = randint(0, len(WORDS) - 1)
    rn2 = randint(0, len(WORDS) - 1)
    
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
    # after a lot of training
    
    sampled_images = diffusion.sample(batch_size = 4)
    sampled_images.shape # (4, 3, 128, 128)
```

Samples and model checkpoints will be logged to `./results_folder` periodically

## Multi-GPU Training

The `Trainer` class is now equipped with <a href="https://huggingface.co/docs/accelerate/accelerator">🤗 Accelerator</a>. You can easily do multi-gpu training in two steps using their `accelerate` CLI

At the project root directory, where the training script is, run

```python
$ accelerate config
```

Then, in the same directory

```python
$ accelerate launch train.py
```

## Miscellaneous

### 1D Sequence

By popular request, a 1D Unet + Gaussian Diffusion implementation.

```python
import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 32
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 128,
    timesteps = 1000,
    objective = 'pred_v'
)

training_seq = torch.rand(64, 32, 128) # features are normalized from 0 to 1

loss = diffusion(training_seq)
loss.backward()

# Or using trainer

dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 32, 128)

```

`Trainer1D` does not evaluate the generated samples in any way since the type of data is not known.

You could consider adding a suitable metric to the training loop yourself after doing an editable install of this package
`pip install -e .`.

## Citations

```bibtex
@inproceedings{NEURIPS2020_4c5bcfec,
    author      = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
    pages       = {6840--6851},
    publisher   = {Curran Associates, Inc.},
    title       = {Denoising Diffusion Probabilistic Models},
    url         = {https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf},
    volume      = {33},
    year        = {2020}
}
```

```bibtex
@InProceedings{pmlr-v139-nichol21a,
    title       = {Improved Denoising Diffusion Probabilistic Models},
    author      = {Nichol, Alexander Quinn and Dhariwal, Prafulla},
    booktitle   = {Proceedings of the 38th International Conference on Machine Learning},
    pages       = {8162--8171},
    year        = {2021},
    editor      = {Meila, Marina and Zhang, Tong},
    volume      = {139},
    series      = {Proceedings of Machine Learning Research},
    month       = {18--24 Jul},
    publisher   = {PMLR},
    pdf         = {http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf},
    url         = {https://proceedings.mlr.press/v139/nichol21a.html},
}
```

```bibtex
@inproceedings{kingma2021on,
    title       = {On Density Estimation with Diffusion Models},
    author      = {Diederik P Kingma and Tim Salimans and Ben Poole and Jonathan Ho},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
    year        = {2021},
    url         = {https://openreview.net/forum?id=2LdBqxc1Yv}
}
```

```bibtex
@article{Karras2022ElucidatingTD,
    title   = {Elucidating the Design Space of Diffusion-Based Generative Models},
    author  = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2206.00364}
}
```

```bibtex
@article{Song2021DenoisingDI,
    title   = {Denoising Diffusion Implicit Models},
    author  = {Jiaming Song and Chenlin Meng and Stefano Ermon},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2010.02502}
}
```

```bibtex
@misc{chen2022analog,
    title   = {Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning},
    author  = {Ting Chen and Ruixiang Zhang and Geoffrey Hinton},
    year    = {2022},
    eprint  = {2208.04202},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@article{Salimans2022ProgressiveDF,
    title   = {Progressive Distillation for Fast Sampling of Diffusion Models},
    author  = {Tim Salimans and Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2202.00512}
}
```

```bibtex
@article{Ho2022ClassifierFreeDG,
    title   = {Classifier-Free Diffusion Guidance},
    author  = {Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2207.12598}
}
```

```bibtex
@article{Sunkara2022NoMS,
    title   = {No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects},
    author  = {Raja Sunkara and Tie Luo},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.03641}
}
```

```bibtex
@inproceedings{Jabri2022ScalableAC,
    title   = {Scalable Adaptive Computation for Iterative Generation},
    author  = {A. Jabri and David J. Fleet and Ting Chen},
    year    = {2022}
}
```

```bibtex
@article{Cheng2022DPMSolverPlusPlus,
    title   = {DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models},
    author  = {Cheng Lu and Yuhao Zhou and Fan Bao and Jianfei Chen and Chongxuan Li and Jun Zhu},
    journal = {NeuRips 2022 Oral},
    year    = {2022},
    volume  = {abs/2211.01095}
}
```

```bibtex
@inproceedings{Hoogeboom2023simpleDE,
    title   = {simple diffusion: End-to-end diffusion for high resolution images},
    author  = {Emiel Hoogeboom and Jonathan Heek and Tim Salimans},
    year    = {2023}
}
```

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.01327,
    doi     = {10.48550/ARXIV.2302.01327},
    url     = {https://arxiv.org/abs/2302.01327},
    author  = {Kumar, Manoj and Dehghani, Mostafa and Houlsby, Neil},
    title   = {Dual PatchNorm},
    publisher = {arXiv},
    year    = {2023},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

```bibtex
@inproceedings{Hang2023EfficientDT,
    title   = {Efficient Diffusion Training via Min-SNR Weighting Strategy},
    author  = {Tiankai Hang and Shuyang Gu and Chen Li and Jianmin Bao and Dong Chen and Han Hu and Xin Geng and Baining Guo},
    year    = {2023}
}
```

```bibtex
@misc{Guttenberg2023,
    author  = {Nicholas Guttenberg},
    url     = {https://www.crosslabs.org/blog/diffusion-with-offset-noise}
}
```

```bibtex
@inproceedings{Lin2023CommonDN,
    title   = {Common Diffusion Noise Schedules and Sample Steps are Flawed},
    author  = {Shanchuan Lin and Bingchen Liu and Jiashi Li and Xiao Yang},
    year    = {2023}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@article{Bondarenko2023QuantizableTR,
    title   = {Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing},
    author  = {Yelysei Bondarenko and Markus Nagel and Tijmen Blankevoort},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2306.12929},
    url     = {https://api.semanticscholar.org/CorpusID:259224568}
}
```

```bibtex
@article{Karras2023AnalyzingAI,
    title   = {Analyzing and Improving the Training Dynamics of Diffusion Models},
    author  = {Tero Karras and Miika Aittala and Jaakko Lehtinen and Janne Hellsten and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2312.02696},
    url     = {https://api.semanticscholar.org/CorpusID:265659032}
}
```

```bibtex
@article{Li2024ImmiscibleDA,
    title   = {Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment},
    author  = {Yiheng Li and Heyang Jiang and Akio Kodaira and Masayoshi Tomizuka and Kurt Keutzer and Chenfeng Xu},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.12303},
    url     = {https://api.semanticscholar.org/CorpusID:270562607}
}
```

```bibtex
@article{Chung2024CFGMC,
    title   = {CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models},
    author  = {Hyungjin Chung and Jeongsol Kim and Geon Yeong Park and Hyelin Nam and Jong Chul Ye},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.08070},
    url     = {https://api.semanticscholar.org/CorpusID:270391454}
}
```
