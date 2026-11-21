#include <cuda_runtime.h>

__global__ void add_kernel(const float *a, const float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" int launch_add(const float *a, const float *b, float *out, int n,
                          cudaStream_t stream) {
    add_kernel<<<(n + 255) / 256, 256, 0, stream>>>(a, b, out, n);
    return static_cast<int>(cudaGetLastError());
}
