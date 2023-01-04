#!/bin/bash
#SBATCH -p dgx-workshop   # partition name
#SBATCH --gres=gpu:1      # use 1 GPU

# set EXEC
if [ -z "$EXEC" ]; then
    EXEC=bin/base
fi

# load modules
module purge
module load cuda-toolkit
module use /cfca-work/gpuws00/opt/modules
module load hdf5
module load boost

# set environmental variables for OpenMP
OMP_OPT_ENV="env OMP_DISPLAY_ENV=verbose OMP_PLACES=cores OMP_PROC_BIND=close" # for GCC or LLVM

# job execution by using SLURM
cd $SLURM_SUBMIT_DIR
$OMP_OPT_ENV numactl --localalloc $EXEC $OPTION

exit 0
