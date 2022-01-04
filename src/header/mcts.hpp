#pragma once

#include <iostream>
#include <vector>
#include "node.hpp"

using namespace std;

class MCTS{
    public:
        static void train(Node& root_node, int simulation);
        static pair<int, int> select_action(Node& root_node);
};