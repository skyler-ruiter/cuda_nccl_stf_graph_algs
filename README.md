# IUB-ISE Spring 2025 Final Project
## Benchmarking Distributed Graph Algorithms with NCCL and CUDASTF
## Skyler Ruiter
### ENGR-E536 High Performance Graph Analytics
### CSCI-P434 Distributed Systems

---

### Instructions:

*BigRed200 Library Testing*

* **Description:** tests a simple program that moves memory between 2 nodes with 2 gpus each using nccl, cudastf, and mpi. 

* Compile Program: `make mpi_nccl_testing`

* Allocate Nodes: `salloc -A r01156 -p gpu-debug --gpus-per-node 2 --nodes 2 --ntasks-per-node=2`

* Load Modules: `module load PrgEnv-nvidia cudatoolkit/12.2 gcc/12.2 nccl cmake`

* Run Code: `srun -N 2 -n 4 ./mpi_nccl_testing`

* Expected Output:

```
sruiter@login1:~/school/cuda_nccl_stf_graph_algs/src> srun -N 2 -n 4 ./mpi_nccl_testing
Rank 0 received: 0.00 0.84 0.91 0.14 -0.76 -0.96 -0.28 0.66 0.99 0.41 -0.54 -1.00 -0.54 0.42 0.99 0.65 10.00 10.84 10.91 10.14 9.24 9.04 9.72 10.66 10.99 10.41 9.46 9.00 9.46 10.42 10.99 10.65 20.00 20.84 20.91 20.14 19.24 19.04 19.72 20.66 20.99 20.41 19.46 19.00 19.46 20.42 20.99 20.65 30.00 30.84 30.91 30.14 29.24 29.04 29.72 30.66 30.99 30.41 29.46 29.00 29.46 30.42 30.99 30.65 
Rank 1 received: 0.00 0.84 0.91 0.14 -0.76 -0.96 -0.28 0.66 0.99 0.41 -0.54 -1.00 -0.54 0.42 0.99 0.65 10.00 10.84 10.91 10.14 9.24 9.04 9.72 10.66 10.99 10.41 9.46 9.00 9.46 10.42 10.99 10.65 20.00 20.84 20.91 20.14 19.24 19.04 19.72 20.66 20.99 20.41 19.46 19.00 19.46 20.42 20.99 20.65 30.00 30.84 30.91 30.14 29.24 29.04 29.72 30.66 30.99 30.41 29.46 29.00 29.46 30.42 30.99 30.65 
Rank 2 received: 0.00 0.84 0.91 0.14 -0.76 -0.96 -0.28 0.66 0.99 0.41 -0.54 -1.00 -0.54 0.42 0.99 0.65 10.00 10.84 10.91 10.14 9.24 9.04 9.72 10.66 10.99 10.41 9.46 9.00 9.46 10.42 10.99 10.65 20.00 20.84 20.91 20.14 19.24 19.04 19.72 20.66 20.99 20.41 19.46 19.00 19.46 20.42 20.99 20.65 30.00 30.84 30.91 30.14 29.24 29.04 29.72 30.66 30.99 30.41 29.46 29.00 29.46 30.42 30.99 30.65 
Rank 3 received: 0.00 0.84 0.91 0.14 -0.76 -0.96 -0.28 0.66 0.99 0.41 -0.54 -1.00 -0.54 0.42 0.99 0.65 10.00 10.84 10.91 10.14 9.24 9.04 9.72 10.66 10.99 10.41 9.46 9.00 9.46 10.42 10.99 10.65 20.00 20.84 20.91 20.14 19.24 19.04 19.72 20.66 20.99 20.41 19.46 19.00 19.46 20.42 20.99 20.65 30.00 30.84 30.91 30.14 29.24 29.04 29.72 30.66 30.99 30.41 29.46 29.00 29.46 30.42 30.99 30.65 
```

---

*BigRed200 Distributed BFS*

* **Description:** Runs a distributed BFS search from a random or CL provided vertex across 4 gpus and 2 nodes, with 2 nodes per gpu and 4 total tasks.

* Will need to allocate gpu nodes to run (see salloc above for gpu-debug example) and load the modules in the previous example.

* Compile Program (in project home):

```
cmake -S . -B build
cd build
make
```

* Run Code: `srun -N 2 -n 4 ./dist_bfs [vertex(optional)]`

---