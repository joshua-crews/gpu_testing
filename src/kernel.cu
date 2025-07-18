extern "C" __global__ void add(int *a, int *b, int *c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

extern "C" __global__ void multiply(int *a, int *b, int *c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] * b[idx];
}

extern "C" __global__ void dot_product(const int *a, const int *b, int *c, int N) {
    extern __shared__ int cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    int temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
    }

    if (cacheIndex == 0) {
        atomicAdd(c, cache[0]);
    }
}
