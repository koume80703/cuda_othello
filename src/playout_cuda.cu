#include "header/playout_cuda.cuh"
#include "header/kernel.cuh"

float playout_cuda(State state)
{
    int start = clock();

    int nElem = 4096;
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

    const int threads_per_block = 1024;

    dim3 block(threads_per_block, 1, 1);
    dim3 grid(nElem / block.x, 1, 1);

    std::random_device rnd;
    int seed = rnd();

    int t0 = clock();
    double elapsed = static_cast<double>(t0 - start) / CLOCKS_PER_SEC * 1000.0;
    printf("memory allocate time: %.3f [ms], ", elapsed);
    t0 = clock();

    kernel<<<grid, block>>>(d_sc, d_result, seed);

    CHECK(cudaDeviceSynchronize());

    int t1 = clock();
    elapsed = static_cast<double>(t1 - t0) / CLOCKS_PER_SEC * 1000.0;
    printf("kernel execute time: %.3f [ms], ", elapsed);
    t1 = clock();

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

    int t2 = clock();
    elapsed = static_cast<double>(t2 - t1) / CLOCKS_PER_SEC * 1000.0;
    double total = static_cast<double>(t2 - start) / CLOCKS_PER_SEC * 1000.0;

    printf("others time: %.3f [ms], total time: %.3f [ms]\n", elapsed, total);

    return sum_result;
}
