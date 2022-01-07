#include "header/playout_cuda.cuh"
#include "header/kernel.cuh"
#include "header/measuring.hpp"

float playout_cuda(State state)
{
    double start, tmp, elapsed;
    start = get_time_msec();

    int nElem = N_PLAYOUT;
    size_t size_sc = sizeof(STATE_CUDA);
    size_t size_result = nElem * sizeof(float);

    STATE_CUDA *h_sc, *d_sc;
    h_sc = (STATE_CUDA *)malloc(size_sc);
    trans_data(state, h_sc);

    CHECK(cudaMalloc((STATE_CUDA **)&d_sc, size_sc));

    float *h_result, *d_result;
    h_result = (float *)malloc(size_result);
    memset(h_result, 0, nElem);
    CHECK(cudaMalloc((float **)&d_result, size_result));

    CHECK(cudaMemcpy(d_sc, h_sc, size_sc, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_result, h_result, size_result, cudaMemcpyHostToDevice));

    const int threads_per_block = 512;

    dim3 block(threads_per_block, 1, 1);
    dim3 grid((nElem + block.x -1) / block.x, 1, 1);

    std::random_device rnd;
    int seed = rnd();

    // memory allocate time
    tmp = get_time_msec();
    elapsed = tmp - start;
    start = tmp;
    extern double malloc_time;
    malloc_time = elapsed;
    
    kernel<<<grid, block>>>(d_sc, d_result, seed);

    CHECK(cudaDeviceSynchronize());

    // exectution time
    tmp = get_time_msec();
    elapsed = tmp - start;
    start = tmp;
    extern double exe_time;
    exe_time = elapsed;

    CHECK(cudaMemcpy(h_result, d_result, size_result, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_sc));
    CHECK(cudaFree(d_result));

    int sum_result = 0;
    for (int i = 0; i < nElem; i++)
    {
        sum_result += h_result[i];
    }

    free(h_sc);
    free(h_result);
    
    // others time
    tmp = get_time_msec();
    elapsed = tmp - start;
    extern double others_time;
    others_time = elapsed;


    return sum_result;
}
