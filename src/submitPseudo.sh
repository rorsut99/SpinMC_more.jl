#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=3:00:00
#SBATCH --account=rrg-aparamek
srun --nodes 1 julia pseudoSpin.jl