#include "header/measuring.hpp"


double playout_time;

double malloc_time, exe_time, others_time;

double get_time_msec()
{
    return static_cast<double>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count()) / (1000 * 1000);
}

void print_time(string str, double time)
{
    cout << setw(30) << left << str << resetiosflags(ios_base::floatfield) << ": " << time << " [ms]" << endl;
}

void print_data(string str, double data)
{
    cout << setw(30) << left << str << resetiosflags(ios_base::floatfield) << ": " << data << endl;
}
