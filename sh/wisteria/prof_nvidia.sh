#!/bin/bash
#PJM -L rscgrp=lecture-a
#PJM -L gpu=1
#PJM -L elapse=0:15:00
#PJM -g gt00
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
$OMP_OPT_ENV nsys profile --stats=true numactl --localalloc ${EXEC} ${OPTION} --interval=3.125e-2 --finish=3.125e-2

exit 0
