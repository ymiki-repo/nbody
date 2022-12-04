#!/bin/sh

# read input key-value pairs and extract EXEC (with options)
EXEC=""
for arg in "$@"
do
	IFS='=' read -r key val <<< "$arg"
	case $key in
		--wrapper-Nprocs_node) PROCS_PER_NODE=${val};;
		--wrapper-Nprocs_socket) PROCS_PER_SOCKET=${val};;
		--wrapper-mpl_cfg_dir) TMP_DIR=${val};;
		*) EXEC="$EXEC $arg";;
	esac
done

# obtain rank of MPI process
MPI_RANK=${MV2_COMM_WORLD_RANK:=${PMI_RANK:=${OMPI_COMM_WORLD_RANK:=${PMIX_RANK:=0}}}}

# commit Linux commands
LSCPU=/usr/bin/lscpu
SED=/usr/bin/sed

# set number of MPI processes per node (if necessary)
if [ -z "$PROCS_PER_NODE" ]; then
	CORES_PER_NODE=`LANG=C $LSCPU | $SED -n 's/^CPU(s): *//p'`
	MPI_SIZE=${MV2_COMM_WORLD_SIZE:=${PMI_SIZE:=${OMPI_COMM_WORLD_SIZE:=${OMPI_UNIVERSE_SIZE:=0}}}}
	PROCS_PER_NODE=`expr $MPI_SIZE / $CORES_PER_NODE`
	if [ $PROCS_PER_NODE -lt 1 ]; then
		# when $MPI_SIZE < $CORES_PER_NODE; i.e. a single node can cover all MPI processes
		PROCS_PER_NODE=$MPI_SIZE
	fi
fi

# configuration on NUMA node
if [ `which numactl` ]; then
	DOMAINS_PER_NODE=`LANG=C $LSCPU | $SED -n 's/^NUMA node(s): *//p'`
	SOCKETS_PER_NODE=`LANG=C $LSCPU | $SED -n 's/^Socket(s): *//p'`
	DOMAINS_PER_SOCKET=`expr $DOMAINS_PER_NODE / $SOCKETS_PER_NODE`

	# set number of MPI processes per socket (if necessary)
	if [ -z "$PROCS_PER_SOCKET" ]; then
		PROCS_PER_SOCKET=`expr $PROCS_PER_NODE / $SOCKETS_PER_NODE`
		if [ $PROCS_PER_SOCKET -lt 1 ]; then
			# when $PROCS_PER_NODE < $SOCKETS_PER_NODE; i.e. a single socket can cover all MPI processes in this node
			PROCS_PER_SOCKET=$PROCS_PER_NODE
		fi
	fi

    TEMPID=`expr $MPI_RANK % $PROCS_PER_NODE`
    SOCKET=`expr $TEMPID / $PROCS_PER_SOCKET`
    NUMAID=`expr $SOCKET \* $DOMAINS_PER_SOCKET`
    # NUMACTL="numactl --cpunodebind=$NUMAID --localalloc"
    NUMACTL="numactl --localalloc"
fi

# configuration on NUMA node
if [ `which numactl` ]; then
    TEMPID=`expr $MPI_RANK % $PROCS_PER_NODE`
    SOCKET=`expr $TEMPID / $PROCS_PER_SOCKET`
    # NUMACTL="numactl --cpunodebind=$SOCKET --localalloc"
    NUMACTL="numactl --localalloc"
fi

# set tex cache directory for matplotlib
MPL_CFG_DIR=${TMP_DIR}/p${$}_r${MPI_RANK}
if [ ! -e $MPL_CFG_DIR ]; then
	mkdir -p $MPL_CFG_DIR
fi

# execute job with numactl --localalloc
echo "MPLCONFIGDIR=$MPL_CFG_DIR $NUMACTL $EXEC"
MPLCONFIGDIR=$MPL_CFG_DIR $NUMACTL $EXEC

rm -rf $MPL_CFG_DIR

exit 0
