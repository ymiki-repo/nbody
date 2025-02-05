#!/bin/bash
#PBS -q lecture-mig
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -W group_list=gt00

ROOT_DIR=/work/gt00/$USER

# load system-provided modules
module purge
module load nvidia
module load nv-hpcx
module load hdf5

# setup Python environment
module use ${ROOT_DIR}/opt/modules
module load anyenv
module load miniforge3

# setup Julia environment
module use /work/share/opt/modules/util
module load julia
export JULIA_DEPOT_PATH=${ROOT_DIR}/$(uname -m)/.julia
module load texlive

cd ${PBS_O_WORKDIR}
LD_LIBRARY_PATH=$OMPI_LIB:$LD_LIBRARY_PATH julia jl/plot/error.jl --png ${OPTION}

exit 0
