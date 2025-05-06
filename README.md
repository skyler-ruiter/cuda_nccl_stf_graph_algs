# IUB-ISE Spring 2025 Final Project
## Benchmarking Distributed Graph Algorithms with NCCL and CUDASTF
## Skyler Ruiter
### ENGR-E536 High Performance Graph Analytics
### CSCI-P434 Distributed Systems

---

### Codebase Notes:

* Tried to get all the code working in time but couldn't get it working in time

* I'm confident given more time i could implement and optimize 

* Wrote most of it myself but did us AI for some of the boilerplate and debugging

  * provided some of the resources I used in the `resources/` directory

* There are some random testing files in tests, not tests of actual library mostly library and sanity checks

---

### Instructions:

*BigRed200 Library Testing*

* **Description:** tests a simple program that moves memory between 2 nodes with 2 gpus each using nccl, cudastf, and mpi. 

* Compile Program: `make mpi_nccl_testing`

* Allocate Nodes: `salloc -A r01156 -p gpu-debug --gpus-per-node 2 --nodes 2 --ntasks-per-node=2`

* Load Modules: `module load PrgEnv-nvidia cudatoolkit/12.2 gcc/12.2 nccl cmake`

* Run Code: `srun -N 2 -n 4 ./mpi_nccl_testing`

---

*BigRed200 Distributed BFS*

* **Description:** Runs a distributed BFS search from a random or CL provided vertex across 4 gpus and 2 nodes, with 2 nodes per gpu and 4 total tasks.

* Will need to allocate gpu nodes to run (see salloc above for gpu-debug example) and load the modules in the previous example.

* default vertex and dataset are provided

* **STF Version not currently working** (use non-stf version to compare against cpu)

* Compile Programs (in project home):

``` 
cmake -S . -B build
cd build
make
mpicxx -o cpu_bfs ../src/dist_bfs/cpu_dist_bfs.cpp
```

* Run Distributed BFS (MPI+NCCL+STF) Code: `srun -N 2 -n 4 ./dist_bfs [vertex] [dataset]` 

* Run Distributed BFS (MPI+NCCL): `srun -N 2 -n 4 ./no_stf_dist_bfs [vertex] [dataset]`

* Run Distributed BFS (MPI): `srun -N 2 -n 4 ./cpu_bfs [vertex] [dataset]`

* Run Benchmarks: `sbatch benchmarking/run_tests.sh`

---

*BigRed200 Distributed Pagerank*

* **Description:** Runs a distributed BFS search from a random or CL provided vertex across 4 gpus and 2 nodes, with 2 nodes per gpu and 4 total tasks.

* Will need to allocate gpu nodes to run (see salloc above for gpu-debug example) and load the modules in the previous example.

* default datasets are provided

* **Not currently working (ANY VERSION)** 

* Compile Programs (in project home):

``` 
cmake -S . -B build
cd build
make
mpicxx -o cpu_bfs ../src/dist_bfs/cpu_pagerank.cpp
```

* Run Distributed BFS (MPI+NCCL+STF) Code: `srun -N 2 -n 4 ./dist_pagerank [dataset]` 

* Run Distributed BFS (MPI+NCCL): `srun -N 2 -n 4 ./no_stf_dist_pagerank [dataset]`

* Run Distributed BFS (MPI): `srun -N 2 -n 4 ./cpu_pagerank [dataset]`

---

### Data:

* Graph500 Scale Datasets: https://networkrepository.com/graph500.php

  * Tested scale[18, 19, 21, 23]

* Web UK 2002 Web Graph: https://networkrepository.com/web-uk-2002-all.php

* put in a folder in home directory called `data/` with any other edgelists

  * may need to adapt codebases to read in data formats correctly (run web-uk for example)