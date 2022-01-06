#pragma once

#include <iostream>
#include "state.hpp"

using namespace std;

class Node
{
public:
    Node(State s, int eb);

    float evaluate();
    void expand();
    Node &next_child_based_ucb();
    static float playout(State state);

    State get_state();
    vector<Node> get_children();
    int get_n();

private:
    State state;
    float w;
    int n, expand_base;
    vector<Node> children;
};
