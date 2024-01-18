#!/usr/bin/env bash

set -x

FILE=$1
CONFIG=$2
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
MASTER_PORT=${MASTER_PORT:-$((28500 + $RANDOM % 2000))}
PARTITION=${PARTITION:-DUMMY}
JOB_NAME=${JOB_NAME:-DUMMY}
QUOTATYPE=${QUOTATYPE:-auto}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CUDA_HOME=$(dirname $(dirname $(which nvcc))) \
MASTER_PORT=$MASTER_PORT \
srun -p ${PARTITION} \
  --job-name=${JOB_NAME} \
  --gres=gpu:${GPUS_PER_NODE} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTATYPE} \
  ${SRUN_ARGS} \
  python -u tools/${FILE}.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
