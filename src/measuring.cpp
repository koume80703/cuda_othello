#include "header/measuring.hpp"


double playout_time;

double malloc_time, exe_time, others_time;

double get_time_msec()
{
    return static_cast<double>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count()) / (1000 * 1000);
}

void print_time(string str, double time)
{
    printf("%-30s: %.2f [ms]\n", str.c_str(), time);
}

void print_data(string str, double data)
{
    printf("%-30s: %.2f\n", str.c_str(), data);
}

void print_percentage(double mt, double et, double ot)
{
    printf("percentage -> malloc: %.3f [%%], execution: %.3f [%%], others: %.3f [%%]\n", mt, et, ot);
}
