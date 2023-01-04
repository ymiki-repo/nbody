#!/bin/bash
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=0:15:00
#PJM -g ${PROJ}

ROOT_DIR=/work/${PROJ}/$USER

# load system-provided modules
module purge
module load julia
module load gcc
module load ompi

# setup Python environment
module use ${ROOT_DIR}/opt/modules
module load anyenv
module load miniforge3

# set environmental variables for Julia
export JULIA_DEPOT_PATH=${ROOT_DIR}/.julia

julia jl/package.jl
julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()'

exit 0
