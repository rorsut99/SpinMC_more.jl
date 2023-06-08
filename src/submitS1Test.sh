#!/bin/bash 
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40 
#SBATCH --time=20:00:00 
#SBATCH --account=rrg-aparamek
srun --nodes 2 julia testingTriangularLattice.jl
