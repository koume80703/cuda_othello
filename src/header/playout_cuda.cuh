#pragma once

#include "trans_data.hpp"
#include <random>
#include <chrono>

using namespace std::chrono;

#define CHECK(call)                                       \
    do                                                    \
    {                                                     \
        const cudaError_t error = call;                   \
        if (error != cudaSuccess)                         \
        {                                                 \
            printf("Error: %s:%d, ", __FILE__, __LINE__); \
            printf("code:%d, reason: %s\n", error,        \
                   cudaGetErrorString(error));            \
            exit(1);                                      \
        }                                                 \
    } while (0);

float playout_cuda(State state, PLAYER base_player);
