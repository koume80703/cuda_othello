#include "header/main.hpp"
#include "header/measuring.hpp"

int play_othello()
{
    Game game;
    State state(game);
    while (true)
    {
        if (state.is_done())
        {
            if (BASIC_OUTPUT)
            {
                state.game.show_board();
                state.game.show_score();
            }
            if (state.is_draw())
            {
                return 0;
            }
            else if (state.is_lose())
            {
                return -1;
            }
            else
            {
                return 1;
            }
        }

        if (state.legal_actions().empty())
        {
            state = state.pass_moving();
            continue;
        }

        pair<int, int> action;
        if (state.is_first_player())
        {
            int simulation = N_SIMULATION;
            Node root_node = Node(state, EXPAND_BASE);
            MCTS::train(root_node, simulation);
            action = MCTS::select_action(root_node);
            state = state.next(action);
        }
        else
        {
            action = state.random_action();
            state = state.next(action);
        }
    }
}

int main(void)
{    
    int win = 0, lose = 0, draw = 0;
    const int play_num = 10;

    
    time_t time = system_clock::to_time_t(system_clock::now());
    cout << "Starting program at " << ctime(&time) << endl;
    cout << "CUDA_PLAYOUT: " << CUDA_PLAYOUT << ", N_PLAYOUT: " << N_PLAYOUT << endl;
    cout << "N_SIMULATION: " << N_SIMULATION << ", EXPAND_BASE: " << EXPAND_BASE << endl << endl;

    for (int i = 0; i < play_num; i++)
    {
        cout << "play: " << i << endl;
        int result = play_othello();
        cout << endl;
        if (result == 0)
            draw++;
        else if (result == 1)
            win++;
        else
            lose++;
    }

    if (BASIC_OUTPUT)
    {
        cout << "win: " << win << endl;
        cout << "lose: " << lose << endl;
        cout << "draw: " << draw << endl;
    }

    cout << endl;

    time = system_clock::to_time_t(system_clock::now());
    cout << "Program end at " << ctime(&time) << endl;
}
