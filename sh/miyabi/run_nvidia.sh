#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -W group_list=gz00

# set EXEC
if [ -z "${EXEC}" ]; then
    EXEC=bin/acc_data
fi

# load modules
module purge
module load nvidia
module load hdf5

# job execution
cd ${PBS_O_WORKDIR}
${EXEC} ${OPTION}

exit 0
