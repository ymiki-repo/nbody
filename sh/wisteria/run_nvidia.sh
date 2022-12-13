#!/bin/bash
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=0:30:00
#PJM -g gz00
#PJM -s

# set EXEC
if [ -z "${EXEC}" ]; then
    EXEC=bin/base
fi

# load modules
module purge
module load nvidia
module load hdf5

# set environmental variables for OpenMP
OMP_OPT_ENV="env OMP_DISPLAY_ENV=verbose OMP_PLACES=cores OMP_PROC_BIND=close" # for GCC or LLVM

# job execution
cd $PJM_O_WORKDIR
$OMP_OPT_ENV numactl --localalloc ${EXEC} ${OPTION}

exit 0
