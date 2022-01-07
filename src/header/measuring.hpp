#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>

#define CUDA_PLAYOUT false
#define N_PLAYOUT 512

using namespace std;
using namespace std::chrono;

double get_time_msec();
void print_time(string str, double time);
void print_data(string str, double time);

