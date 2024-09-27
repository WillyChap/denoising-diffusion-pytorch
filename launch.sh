#!/bin/bash
#PBS -N QTTrainDiffusion
#PBS -A P54048000
#PBS -l walltime=12:00:00
#PBS -o CESM_TrainDiffusion.out
#PBS -e CESM_TrainDiffusion.out
#PBS -q main
#PBS -l select=1:ncpus=128:mem=470GB:ngpus=4 -l gpu_type=a100
#PBS -m a
#PBS -M wchapman@ucar.edu

# qsub -I -q main -A P54048000 -l walltime=12:00:00 -l select=1:ncpus=32:mem=470GB:ngpus=4 -l gpu_type=a100

#accelerate config
module load conda
conda activate LuRain
accelerate launch Train_CESM_Conditioned.py
