#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <ctime>

#define CUDA_PLAYOUT false
#define N_PLAYOUT 256
#define BASIC_OUTPUT false

using namespace std;
using namespace std::chrono;

double get_time_msec();
void print_time(string str, double time);
void print_data(string str, double time);
void print_percentage(double mt, double et, double ot);

