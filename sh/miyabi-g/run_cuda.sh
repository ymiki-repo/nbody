#!/bin/bash
#PBS -q lecture-g
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -W group_list=gt00

# set EXEC
if [ -z "${EXEC}" ]; then
    EXEC=bin/cuda_memcpy_shmem
fi

# load modules
module purge
module load cuda
module use /work/share/opt/modules/lib
module load hdf5

# job execution
cd ${PBS_O_WORKDIR}
${EXEC} ${OPTION}

exit 0
