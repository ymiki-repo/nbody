#!/bin/bash
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM --mpi proc=9
#PJM -L elapse=0:30:00
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
module load miniconda3

# set environmental variables for Julia
export JULIA_DEPOT_PATH=${ROOT_DIR}/.julia

# for texlive
export HOME=${ROOT_DIR}

# set tmp for matplotlib
MPL_CFG_DIR=/tmp/matplotlib

# set number of MPI processes per node
# tentative implementation: this script assumes single node execution
MPI_PROC_NODE=${PJM_MPI_PROC}

# set number of MPI processes per socket
# tentative implementation: this script assumes single socket execution
MPI_PROC_SOCKET=${PJM_MPI_PROC}

# --png option is NOT work on Wisteria/BDEC-01 (dvipng is missing)
mpiexec -machinefile ${PJM_O_NODEINF} -n ${PJM_MPI_PROC} sh/wrapper/mpi_matplotlib.sh --wrapper-Nprocs_node=${MPI_PROC_NODE} --wrapper-Nprocs_socket=${MPI_PROC_SOCKET} --wrapper-mpl_cfg_dir=$MPL_CFG_DIR julia jl/plot/dot.jl --svg ${OPTION}

exit 0
