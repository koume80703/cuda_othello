#include "trans_data.hpp"

#include <curand_kernel.h>

#define LA_SIZE 100
#define FLIPPABLE_SIZE 100
#define START_PLAYER 1

__global__ void kernel(STATE_CUDA *sc, float *result, int seed);

__device__ float playout_gpu(STATE_CUDA *sc, curandState *s);

__device__ bool is_done(STATE_CUDA *sc);
__device__ bool is_draw(STATE_CUDA *sc);
__device__ bool is_win(STATE_CUDA *sc);
__device__ bool is_lose(STATE_CUDA *sc);

__device__ void legal_actions(STATE_CUDA *sc, int la[][2]);
__device__ void flippable_stone(STATE_CUDA *sc, int x, int y, int flippable[][2]);
__device__ bool next(STATE_CUDA *sc, int x, int y);
__device__ bool set_stone(STATE_CUDA *sc, int x, int y);
__device__ void pass_moving(STATE_CUDA *sc);
__device__ void finish_game(STATE_CUDA *sc);
__device__ int random_action(STATE_CUDA *sc, int la[][2], curandState *s);
