#!/bin/bash
#SBATCH --job-name=morpho    # create a short name for your job
#SBATCH -o logs/morphoGb.sh.log-%j
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH -c 30                     # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=36:00:00

module load julia
module load hdf5

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
julia -p 30 run_biggrid.jl