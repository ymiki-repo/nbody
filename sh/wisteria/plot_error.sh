#!/bin/bash
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=0:30:00
#PJM -g ${PROJ}

ROOT_DIR=/work/${PROJ}/$USER

# load system-provided modules
module purge
module load julia

# setup Python environment
module use ${ROOT_DIR}/opt/modules
module load anyenv
module load miniconda3

# set environmental variables for Julia
export JULIA_DEPOT_PATH=${ROOT_DIR}/.julia

# for texlive
export HOME=${ROOT_DIR}

# --png option is NOT work on Wisteria/BDEC-01 (dvipng is missing)
numactl --localalloc julia jl/plot/error.jl --svg ${OPTION}

exit 0
