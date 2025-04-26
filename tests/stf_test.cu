#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

template <typename T>
__global__ void axpy(T a, slice<T> x, slice<T> y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;

    for (int ind = tid; ind < x.size(); ind += nthreads) {
        y(ind) += a * x(ind);
    }
}

int main(int argc, char** argv) {
    context ctx;

    const size_t N = 16;
    double X[N], Y[N];

    for (size_t ind = 0; ind < N; ind++) {
        X[ind] = sin((double)ind);
        Y[ind] = col((double)ind);
    }

    auto lX = ctx.logical_data(X);
    auto lY = ctx.logical_data(Y);

    double alpha = 3.14;

    /* Compute Y = Y + alpha X */
    ctx.task(lX.read(), lY.rw())->*[&](cudaStream_t s, auto sX, auto sY) {
        axpy<<<16, 128, 0, s>>>(alpha, sX, sY);
    };

    ctx.finalize();
}