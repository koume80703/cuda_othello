#include "header/node.hpp"
#include "header/ucb1.hpp"
#include "header/argmax.hpp"
#include "header/playout_cuda.cuh"
#include "header/measuring.hpp"

Node::Node(State s, int eb = 20)
{
    state = s;

    w = 0;
    n = 0;
    expand_base = eb;
    children = vector<Node>();
}

float Node::evaluate()
{
    if (state.is_done())
    {
        float value = (state.is_lose()) ? -1 : 0;
        w += value;
        n++;
        return value;
    }

    if (children.empty())
    {
        State tmp = state;

        float value = 0;
        double start, end, elapsed;
        if (CUDA_PLAYOUT)
        {
            start = get_time_msec();
            
            value += playout_cuda(tmp);
            
            end = get_time_msec();
            elapsed = end - start;
            print_time("elapsed time with CUDA", elapsed);
            
            extern double malloc_time, exe_time, others_time;
            double mt_per_total, et_per_total, ot_per_total;
            mt_per_total = malloc_time * 100 / elapsed;
            et_per_total = exe_time * 100 / elapsed;
            ot_per_total = others_time * 100 / elapsed;
            print_percentage(mt_per_total, et_per_total, ot_per_total);
        }
        else
        {
            start = get_time_msec();
            
            for (int i = 0; i < N_PLAYOUT; i++)
            {
                value += Node::playout(tmp);
            }
            
            end = get_time_msec();
            elapsed = end - start;
            print_time("elapsed time with CPU", elapsed);
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
        children.push_back(Node(state.next(action), expand_base));
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

float Node::playout(State state)
{
    if (state.is_done())
    {
        if (state.is_win())
        {
            return 1;
        }
        else if (state.is_lose())
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
        return Node::playout(state);
    }
    else
    {
        return Node::playout(state.next(state.random_action()));
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
