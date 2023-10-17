#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --time=6:00:00
#SBATCH --account=rrg-aparamek

direc="pseudoSpinMCdatT=${T}"

mkdir -p $direc
cp pseudoSpin.jl $direc
cd $direc

srun --nodes 4 julia pseudoSpin.jl ${T}

