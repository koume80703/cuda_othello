#include "../header/ucb1.hpp"

float ucb1(int sn, int n, float w){
    return w / n + sqrt((2 * std::log(sn) / n));
}