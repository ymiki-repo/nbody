#!/bin/sh
set -e

# read input key-value pairs and ARGVs for the executable
ARGV=""
for arg in "$@"
do
	IFS='=' read -r key val <<< "$arg"
	case $key in
		--file) FILE=${val};;
		--base) BASE=${val};;
		--exec) EXEC=${val};;
		--job-exec) JOB_SCRIPT_EXEC=${val};;
		--job-plot) JOB_SCRIPT_PLOT=${val};;
		--julia-plot) JULIA_PLOT=${val};;
		*) ARGV="$ARGV ${arg}";;
	esac
done

# name of the simulation
if [ -z "$FILE" ]; then
    FILE=benchmark
fi
COMMON_OPTION="--file=${FILE} ${ARGV}"

INPUT="--export="
# module name of the base compiler
if [ -n "$BASE" ]; then
    INPUT="${INPUT}BASE=${BASE},"
fi

# name of the executable
if [ -n "$EXEC" ]; then
    INPUT="${INPUT}EXEC=${EXEC},"
fi

# name of the job script for execution
if [ -z "$JOB_SCRIPT_EXEC" ]; then
    JOB_SCRIPT_EXEC=sh/slurm/run.sh
fi

# name of the job script for visualization
if [ -z "$JOB_SCRIPT_PLOT" ]; then
    JOB_SCRIPT_PLOT=sh/slurm/plot_serial.sh
fi

# name of the Julia script for visualization
if [ -z "$JULIA_PLOT" ]; then
    JULIA_PLOT=jl/plot/scaling_error.jl
fi

for NUM in 256
do
	for ETA in 0.5 0.25 0.125 6.25e-2 3.125e-2 1.5625e-2 7.8125e-3 3.90625e-3 1.953125e-3
	do
		OPTION="OPTION=\"${COMMON_OPTION} --num=${NUM} --eta=${ETA}\""
		RET=$(eval "sbatch --dependency=singleton ${INPUT}${OPTION} ${JOB_SCRIPT_EXEC}")
	done
done

JOB_ID=`echo ${RET} | awk '{print $4}'`
sbatch --dependency=afterany:${JOB_ID} --export=EXEC="julia ${JULIA_PLOT} --target=${FILE}" ${JOB_SCRIPT_PLOT}

exit 0
