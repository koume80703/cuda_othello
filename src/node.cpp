#include "header/node.hpp"
#include "header/ucb1.hpp"
#include "header/argmax.hpp"
#include "header/playout_cuda.cuh"
#include "header/measuring.hpp"

Node::Node(State s, PLAYER bp, int eb = 20)
{
    state = s;

    w = 0;
    n = 0;
    expand_base = eb;
    children = vector<Node>();
    base_player = bp;
}

float Node::evaluate()
{
    if (state.is_done())
    {
        float value = (state.is_lose(base_player)) ? -1 : 0;
        w += value;
        n++;
        return value;
    }

    if (children.empty())
    {
        State tmp = state;

        float value = 0;

        double start, end, elapsed_cuda, elapsed_cpu;
        extern int n_playout;
        if (CUDA_PLAYOUT)
        {
            if (TIME_OUTPUT)
            {
                start = get_time_msec();
            }
            
            value += playout_cuda(tmp, base_player);

            if (TIME_OUTPUT)
            {
                end = get_time_msec();
                elapsed_cuda = end - start;

                if (!ONLY_CUDA)
                {
                    start = get_time_msec();
                    for (int i = 0; i < n_playout; i++)
                    {
                        Node::playout(tmp, base_player);
                    }
                    end = get_time_msec();
                    elapsed_cpu = end - start;
                }
            }
        }
        else
        {
            if (TIME_OUTPUT)
            {
                start = get_time_msec();
            }
            
            for (int i = 0; i < n_playout; i++)
            {
                value += Node::playout(tmp, base_player);
            }
            
            if (TIME_OUTPUT)
            {
                end = get_time_msec();
                elapsed_cpu = end - start;
                
                start = get_time_msec();
                playout_cuda(tmp, base_player);
                end = get_time_msec();
                elapsed_cuda = end - start;
            }
        }

        if (TIME_OUTPUT)
        {
            extern double total_cuda, total_cpu;
            total_cuda += elapsed_cuda;
            total_cpu += elapsed_cpu;

            /*
            extern double malloc_time, exe_time, others_time;
            printf("elapsed, %.3f\n", elapsed_cuda);
            print_percentage(malloc_time/elapsed_cuda, exe_time/elapsed_cuda, others_time/elapsed_cuda);            
            */
        }
        
        w += value;
        n++;

        if (n == expand_base)
        {
            expand();
        }

        return value;
    }
    else
    {
        Node &child = next_child_based_ucb();
        float value = child.evaluate();
        w += value;
        n++;

        return value;
    }
}

void Node::expand()
{
    if (state.legal_actions().empty())
    {
        return;
    }
    for (auto action : state.legal_actions())
    {
        children.push_back(Node(state.next(action), base_player, expand_base));
    }
}

Node &Node::next_child_based_ucb()
{
    for (auto &child : children)
    {
        if (child.n == 0)
        {
            return child;
        }
    }

    int sum_n = 0;
    for (auto &child : children)
    {
        sum_n += child.n;
    }

    vector<float> ucb1_values = vector<float>();
    for (auto &child : children)
    {
        float ucb = ucb1(sum_n, child.n, child.w);
        ucb1_values.push_back(ucb);
    }
    return children.at(argmax(ucb1_values));
}

float Node::playout(State state, PLAYER base_player)
{
    if (state.is_done())
    {
        if (state.is_win(base_player))
        {
            return 1;
        }
        else if (state.is_lose(base_player))
        {
            return -1;
        }
        else
        {
            return 0;
        }
    }
    if (state.legal_actions().empty())
    {
        state = state.pass_moving();
        return Node::playout(state, base_player);
    }
    else
    {
        return Node::playout(state.next(state.random_action()), base_player);
    }
}

State Node::get_state()
{
    return state;
}

vector<Node> Node::get_children()
{
    return children;
}

int Node::get_n()
{
    return n;
}
