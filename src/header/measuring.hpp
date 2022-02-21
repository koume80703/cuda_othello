#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <ctime>

#define CUDA_PLAYOUT true
#define ONLY_CUDA true
#define BASIC_OUTPUT false
#define TIME_OUTPUT true
#define MESURING_STRENGTH false

#define N_PLAYOUT 2048
#define N_SIMULATION 100
#define EXPAND_BASE 20
#define THREADS_PER_BLOCK 512


using namespace std;
using namespace std::chrono;

double get_time_msec();
void print_time(string str, double time);
void print_data(string str, double time);
void print_percentage(double mt, double et, double ot);

