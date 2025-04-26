# IUB-ISE Spring 2025 Final Project
## Benchmarking Distributed Graph Algorithms with NCCL and CUDASTF
## Skyler Ruiter
### ENGR-E536 High Performance Graph Analytics
### CSCI-P434 Distributed Systems

---

### Instructions:

*BigRed200 Library Testing*

* Compile Program: `make mpi_nccl_testing`

* Allocate Nodes: `salloc -A r01156 --exclusive -p gpu-debug --gpus-per-node 2 --nodes 2`

* Load Modules: `module load PrgEnv-nvidia cudatoolkit/12.2 gcc/12.2 nccl cmake`

* Run Code: `srun -N 2 -n 4 ./mpi_nccl_testing`

* Expected Output:

```
sruiter@login1:~/school/cuda_nccl_stf_graph_algs/src> srun -N 2 -n 4 ./mpi_nccl_testing
Rank 2 received: 0.0 1.0 2.0 3.0 10.0 11.0 12.0 13.0 20.0 21.0 22.0 23.0 30.0 31.0 32.0 33.0 
Rank 0 received: 0.0 1.0 2.0 3.0 10.0 11.0 12.0 13.0 20.0 21.0 22.0 23.0 30.0 31.0 32.0 33.0 
Rank 3 received: 0.0 1.0 2.0 3.0 10.0 11.0 12.0 13.0 20.0 21.0 22.0 23.0 30.0 31.0 32.0 33.0 
Rank 1 received: 0.0 1.0 2.0 3.0 10.0 11.0 12.0 13.0 20.0 21.0 22.0 23.0 30.0 31.0 32.0 33.0 
```