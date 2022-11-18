#!/bin/sh
#SBATCH -J run_cpu   # name of job
#SBATCH -p regular   # partition name

# set EXEC
if [ -z "$EXEC" ]; then
    EXEC=bin/base
fi

# load modules
module purge
if [ -n "$BASE" ]; then
    module load $BASE
fi
module load hdf5
module load boost

# set environmental variables for OpenMP
if [ "$BASE" == intel ]; then
	OMP_OPT_ENV="env KMP_AFFINITY=verbose,granularity=core,balanced" # for Intel Compilers
else
	OMP_OPT_ENV="env OMP_DISPLAY_ENV=verbose OMP_PLACES=cores OMP_PROC_BIND=close" # for GCC or LLVM
fi

# job execution by using SLURM
if [ `which numactl` ]; then
	echo "$OMP_OPT_ENV numactl --localalloc $EXEC $OPTION"
	$OMP_OPT_ENV numactl --localalloc $EXEC $OPTION
else
	echo "$OMP_OPT_ENV $EXEC $OPTION"
	$OMP_OPT_ENV $EXEC $OPTION
fi

exit 0
