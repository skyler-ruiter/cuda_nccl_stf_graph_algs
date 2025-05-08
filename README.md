# IUB-ISE Spring 2025 Final Project
## Benchmarking Distributed Graph Algorithms with NCCL and CUDASTF
## Skyler Ruiter
### ENGR-E536 High Performance Graph Analytics
### CSCI-P434 Distributed Systems

---

### Codebase Notes:

* `benchmarking/`: has output from benchmarking (look for more details about runtime info), scripts for running tests and manipuating data, and output plots. 

  * `plots/`: has cool plots, but there's also `task_graphs/`, which has some of the task graphs of the CUDASTF version (some colored by relative times, main path highlighted with red circles, some with whole graph run some cut off after 2 levels)

* `resources/`: has some files I used to base the code off of, and some quick notes about the codebase.

* `tests`: has a few testing files to look at graph analytics and run some basic library testing.

  * there is a file with the 10 highest and lowest vertex degrees for the datasets used in the benchmarking

* `src`: has the implementations of the distributed graph algorithms. 

  * `dist_bfs`: has implementations of Breadth First Search (BFS) using MPI, OpenMP+MPI, NCCL+MPI, and CUDASTF+NCCL+MPI. 

  * `pagerank`: has implementations of distributed pagerank -- NOTE: WAS NOT ABLE TO COMPLETE

--

* Wrote most of it myself but did use AI for some of the boilerplate and debugging

* tried a lot of methods and failed a lot, but proud of what I was able to implement. 

* felt like i learned a lot abuot using NCCL and using distributed systems and how much a pain graph algorithms are. 

---

### Instructions:

*BigRed200 Library Testing*

* **Description:** tests a simple program that moves memory between 2 nodes with 2 gpus each using nccl, cudastf, and mpi. 

* Compile Program: `make mpi_nccl_testing`

* Allocate Nodes: `salloc -A r01156 -p gpu-debug --gpus-per-node 2 --nodes 2 --ntasks-per-node=2`

* Load Modules: `module load PrgEnv-nvidia cudatoolkit/12.2 gcc/12.2 nccl cmake python`

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
mpicxx -O3 -o cpu_bfs_no_openmp ../src/dist_bfs/cpu_dist_bfs.cpp
mpicxx -O3 -fopenmp -o cpu_bfs ../src/dist_bfs/cpu_dist_bfs_openmp.cpp
```

* Run Distributed BFS (MPI+NCCL+STF) Code: `srun -N 2 -n 4 ./dist_bfs [vertex] [dataset]` 

* Run Distributed BFS (MPI+NCCL): `srun -N 2 -n 4 ./no_stf_dist_bfs [vertex] [dataset]`

* Run Distributed BFS (MPI+OpenMP): `srun -N 2 -n 4 ./cpu_bfs [vertex] [dataset]`

* Run Distributed BFS (MPI): `srun -N 2 -n 4 ./cpu_bfs_no_openmp [vertex] [dataset]`

* Run Benchmarks: `sbatch benchmarking/run_tests.sh`

  * DOESNT TEST MPI+OPENMP VERSION (due to bad performance compared to single threaded)

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

  * Tested scale[18, 19, 21, 23] -- all undirected (leading to less graph discovery)

* Web UK 2002 Web Graph: https://networkrepository.com/web-uk-2002-all.php

  * directed web network graph

* put in a folder in home directory called `data/` with any other edgelists

  * may need to adapt codebases to read in data formats correctly (run web-uk for example)