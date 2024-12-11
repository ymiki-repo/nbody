#!/bin/bash
#PBS -q debug-mig
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -W group_list=gz00
#PBS -j oe

# load modules
module purge
module load nvidia
module load hdf5

cd ${PBS_O_WORKDIR}
ninja

exit 0
