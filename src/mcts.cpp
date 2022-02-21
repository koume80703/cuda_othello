#include "header/mcts.hpp"
#include "header/state.hpp"
#include "header/argmax.hpp"
#include "header/measuring.hpp"

void MCTS::train(Node &root_node, int simulation)
{
    root_node.expand();

    extern double total_cuda, total_cpu;
    if (TIME_OUTPUT)
    {
        total_cuda = 0;
        total_cpu = 0;
    }
    
    for (int i = 0; i < simulation; i++)
    {
        root_node.evaluate();
    }

    if (TIME_OUTPUT)
    {
        printf("%2d, ", root_node.get_state().game.get_turn());
        printf("%7.3f, %7.3f\n", total_cuda / simulation, total_cpu / simulation);
    }
}

pair<int, int> MCTS::select_action(Node &root_node)
{
    vector<pair<int, int>> legal_actions = root_node.get_state().legal_actions();
    vector<int> visit_list;
    for (auto child : root_node.get_children())
    {
        visit_list.push_back(child.get_n());
    }
    return legal_actions.at(argmax(visit_list));
}
