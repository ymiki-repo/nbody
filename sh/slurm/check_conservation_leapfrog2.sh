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
	for DT in 1.5625e-2 7.8125e-3 3.90625e-3 1.953125e-3 9.765625e-4 4.8828125e-4 2.44140625e-4 1.220703125e-4 6.103515625e-5 3.0517578125e-5 1.52587890625e-5 7.62939453125e-6 3.814697265625e-6 1.9073486328125e-6 9.5367431640625e-7
	do
		OPTION="OPTION=\"${COMMON_OPTION} --num=${NUM} --time_step=${DT}\""
		RET=$(eval "sbatch --dependency=singleton ${INPUT}${OPTION} ${JOB_SCRIPT_EXEC}")
	done
done

JOB_ID=`echo ${RET} | awk '{print $4}'`
sbatch --dependency=afterany:${JOB_ID} --export=EXEC="julia ${JULIA_PLOT} --target=${FILE}" ${JOB_SCRIPT_PLOT}

exit 0
