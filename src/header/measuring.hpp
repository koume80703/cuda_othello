#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <ctime>

#define CUDA_PLAYOUT true
#define BASIC_OUTPUT false

#define N_PLAYOUT 1
#define N_SIMULATION 100
#define EXPAND_BASE 20

using namespace std;
using namespace std::chrono;

double get_time_msec();
void print_time(string str, double time);
void print_data(string str, double time);
void print_percentage(double mt, double et, double ot);

