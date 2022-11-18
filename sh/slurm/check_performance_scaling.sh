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
    JOB_SCRIPT_EXEC=sh/slurm/run_cpu.sh
fi

# name of the job script for visualization
if [ -z "$JOB_SCRIPT_PLOT" ]; then
    JOB_SCRIPT_PLOT=sh/slurm/plot_serial.sh
fi

# name of the Julia script for visualization
if [ -z "$JULIA_PLOT" ]; then
    JULIA_PLOT=jl/plot/scaling_elapse.jl
fi

for NUM in 256 512 1024 2048 4096 8192 16384 32768 #65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216
do
	OPTION="OPTION=\"${COMMON_OPTION} --num=${NUM}\""
	RET=$(eval "sbatch --dependency=singleton ${INPUT}${OPTION} ${JOB_SCRIPT_EXEC}")
done

JOB_ID=`echo ${RET} | awk '{print $4}'`
sbatch --dependency=afterany:${JOB_ID} --export=EXEC="julia ${JULIA_PLOT} --target=${FILE}" ${JOB_SCRIPT_PLOT}

exit 0
