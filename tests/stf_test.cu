#include <cuda/experimental/stf.cuh>
#include <cuda_runtime.h>

using namespace cuda::experimental::stf;

template <typename T>
__global__ void axpy(T a, slice<T> x, slice<T> y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;

    for (int ind = tid; ind < x.size(); ind += nthreads) {
        y(ind) += a * x(ind);
    }
}