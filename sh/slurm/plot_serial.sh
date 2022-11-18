#!/bin/sh
#SBATCH -J visualize # name of job
#SBATCH -t 00:15:00  # upper limit of elapsed time
#SBATCH -p regular   # partition name

# load modules
module purge
module load anyenv
module load miniconda3
module load julia
module load texlive

# set EXEC
if [ -z "$EXEC" ]; then
    EXEC="julia jl/plot/error.jl"
fi

if [ `which numactl` ]; then
	echo "numactl --localalloc $EXEC"
	numactl --localalloc $EXEC
else
	echo "$EXEC"
	$EXEC
fi

exit 0
