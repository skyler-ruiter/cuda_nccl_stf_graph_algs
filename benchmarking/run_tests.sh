#! /usr/bin/bash

# SLURM directives
#SBATCH --job-name=dist_graph_prof
#SBATCH --output=dist_graph_prof.out
#SBATCH --error=dist_graph_prof.err
#SBATCH --account=r01156
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --gpus-per-node 2
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:30:00
#SBATCH --exclusive

BASE_DIR=${HOME}/school/cuda_nccl_stf_graph_algs
BUILD_DIR=${BASE_DIR}/temp_build
GPU_BIN=${BUILD_DIR}/dist_bfs
CPU_BIN=${BUILD_DIR}/cpu_bfs
DATA_DIR=${BASE_DIR}/data

# Print environment for debugging
echo "Running on nodes:"
scontrol show hostname $SLURM_JOB_NODELIST
echo "BASE_DIR: ${BASE_DIR}"
echo "BUILD_DIR: ${BUILD_DIR}"
echo "DATA_DIR: ${DATA_DIR}"
echo "Data directory contents:"
ls -la ${DATA_DIR}

module load PrgEnv-nvidia cudatoolkit/12.2 gcc/12.2 nccl cmake

OUTPUT_DIR=${BASE_DIR}benchmarking/output/
mkdir -p ${OUTPUT_DIR}

SOURCE_V=10

# Check if data directory exists and has files
if [ ! -d "${DATA_DIR}" ] || [ -z "$(ls -A ${DATA_DIR})" ]; then
    echo "ERROR: Data directory ${DATA_DIR} doesn't exist or is empty"
    exit 1
fi

# Loop through dataset files
for DATASET in ${DATA_DIR}/*; do
    # get the name of the dataset
    DATASET_NAME=$(basename ${DATASET})
    echo "Running BFS on ${DATASET_NAME}"
    echo "Full dataset path: ${DATASET}"

    # run the CPU BFS
    echo "Running CPU BFS"
    srun -n 4 -N 2 --gpus-per-task=0 ${CPU_BIN} ${SOURCE_V} ${DATASET} > ${OUTPUT_DIR}/${DATASET_NAME}_cpu.out

    # run the GPU BFS
    echo "Running GPU BFS"
    srun -n 4 -N 2 ${GPU_BIN} ${SOURCE_V} ${DATASET} > ${OUTPUT_DIR}/${DATASET_NAME}_gpu.out
done