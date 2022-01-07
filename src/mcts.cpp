#include "header/mcts.hpp"
#include "header/state.hpp"
#include "header/argmax.hpp"
#include "header/measuring.hpp"

void MCTS::train(Node &root_node, int simulation)
{
    root_node.expand();
    
    double start, end, elapsed;
    start = get_time_msec();
    
    for (int i = 0; i < simulation; i++)
    {
        printf("i: %2d -> ", i);
        root_node.evaluate();
    }
    
    end = get_time_msec();
    elapsed = end - start;

    print_time("average evaluate time", elapsed / simulation);
    cout << endl;
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
