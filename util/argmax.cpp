#include "../header/argmax.hpp"

int argmax(vector<int> v){
    const int max_value = *max_element(v.begin(), v.end());    
    vector<int>::iterator itr = find(v.begin(), v.end(), max_value);
    if (itr == v.end()) cout << "Not found error: max_value" << endl;
    const int index = distance(v.begin(), itr);

    return index;
}

int argmax(vector<float> v){
    const float max_value = *max_element(v.begin(), v.end());
    vector<float>::iterator itr = find(v.begin(), v.end(), max_value);
    if (itr == v.end()) cout << "Not found error: max_value" << endl;
    const int index = distance(v.begin(), itr);

    return index;
}