#!/bin/sh
#SBATCH -J visualize # name of job
#SBATCH -t 00:15:00  # upper limit of elapsed time
#SBATCH -p regular   # partition name

# load modules
module purge
module load anyenv
module load miniforge3
module load julia
module load texlive

# set EXEC
if [ -z "$EXEC" ]; then
    EXEC="julia jl/plot/error.jl"
fi

if [ `which numactl` ]; then
	echo "numactl --localalloc $EXEC --png $OPTION"
	numactl --localalloc $EXEC --png $OPTION
else
	echo "$EXEC --png $OPTION"
	$EXEC --png $OPTION
fi

exit 0
