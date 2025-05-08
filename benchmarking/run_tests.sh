#! /usr/bin/bash

# SLURM directives
#SBATCH --job-name=dist_graph_prof
#SBATCH --output=dist_graph_prof.out
#SBATCH --error=dist_graph_prof.err
#SBATCH --account=r01156
#SBATCH --partition=gpu-debug
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --gpus-per-node 2
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:30:00
#SBATCH --exclusive

BASE_DIR=${HOME}/school/cuda_nccl_stf_graph_algs
BUILD_DIR=${BASE_DIR}/temp_build
GPU_BIN_STF=${BUILD_DIR}/dist_bfs
GPU_BIN=${BUILD_DIR}/no_stf_dist_bfs
CPU_BIN=${BUILD_DIR}/cpu_bfs_no_openmp
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

# Create timestamped output directory for better organization
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR=${BASE_DIR}/benchmarking/output/${TIMESTAMP}/
echo "Creating output directory: ${OUTPUT_DIR}"
mkdir -p ${OUTPUT_DIR}

# Check if output directory was created successfully
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: Could not create output directory ${OUTPUT_DIR}"
    exit 1
fi

# Check if output directory is writable
if [ ! -w "${OUTPUT_DIR}" ]; then
    echo "ERROR: Output directory ${OUTPUT_DIR} is not writable"
    exit 1
fi

echo "Output directory created successfully: ${OUTPUT_DIR}"
echo "Output directory contents (should be empty):"
ls -la ${OUTPUT_DIR}

SOURCE_V=8

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

    # Define explicit output file paths
    CPU_OUT="${OUTPUT_DIR}${DATASET_NAME}_cpu.out"
    GPU_OUT="${OUTPUT_DIR}${DATASET_NAME}_gpu.out"
    GPU_STF_OUT="${OUTPUT_DIR}${DATASET_NAME}_gpu_stf.out"
    
    # run the CPU BFS
    echo "Running CPU BFS"
    CMD="srun -n 4 -N 2 --gpus-per-task=0 --output=${CPU_OUT} ${CPU_BIN} ${SOURCE_V} ${DATASET}"
    echo "Running command: ${CMD}"
    eval ${CMD}
    
    # Check if CPU output file was created
    echo "Checking for CPU output file: ${CPU_OUT}"
    if [ -f "${CPU_OUT}" ]; then
        echo "CPU output file created successfully, size: $(du -h ${CPU_OUT} | cut -f1)"
    else
        echo "WARNING: CPU output file was not created"
    fi

    # run the GPU BFS
    echo "Running GPU BFS"
    CMD="srun -n 4 -N 2 --output=${GPU_OUT} ${GPU_BIN} ${SOURCE_V} ${DATASET}"
    echo "Running command: ${CMD}"
    eval ${CMD}
    
    # Check if GPU output file was created
    echo "Checking for GPU output file: ${GPU_OUT}"
    if [ -f "${GPU_OUT}" ]; then
        echo "GPU output file created successfully, size: $(du -h ${GPU_OUT} | cut -f1)"
    else
        echo "WARNING: GPU output file was not created"
    fi

    # run the gpu stf BFS
    echo "Running GPU STF BFS"
    CMD="srun -n 4 -N 2 --output=${GPU_STF_OUT} ${GPU_BIN_STF} ${SOURCE_V} ${DATASET}"
    echo "Running command: ${CMD}"
    eval ${CMD}

    # Check if GPU STF output file was created
    echo "Checking for GPU STF output file: ${GPU_STF_OUT}"
    if [ -f "${GPU_STF_OUT}" ]; then
        echo "GPU STF output file created successfully, size: $(du -h ${GPU_STF_OUT} | cut -f1)"
    else
        echo "WARNING: GPU STF output file was not created"
    fi

done

echo "Benchmark completed. All output files should be in: ${OUTPUT_DIR}"
echo "Output directory contents:"
ls -la ${OUTPUT_DIR}