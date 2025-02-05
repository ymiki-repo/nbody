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

# setup Python environment
module use ${ROOT_DIR}/opt/modules
module load anyenv
module load miniforge3

# setup Julia environment
module use /work/share/opt/modules/util
module load julia
export JULIA_DEPOT_PATH=${ROOT_DIR}/$(uname -m)/.julia

cd ${PBS_O_WORKDIR}
julia jl/package.jl
julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary(; library_names=string(Base.ENV["OMPI_LIB"], "/libmpi.so"))'

exit 0
