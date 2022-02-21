#include "header/main.hpp"
#include "header/measuring.hpp"

pair<int,int> mcts_action(State state, PLAYER base_player, int expand_base, int simulation)
{
    Node root_node = Node(state, base_player, expand_base);
    MCTS::train(root_node, simulation);
    return MCTS::select_action(root_node);
}

pair<int, int> random_action(State state)
{
    return state.random_action();
}

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
            else if (state.is_lose(PLAYER::BLACK))
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
            action = mcts_action(state, PLAYER::BLACK, EXPAND_BASE, N_SIMULATION);
            state = state.next(action);
        }
        else
        {
            // action = mcts_action(state, PLAYER::WHITE, EXPAND_BASE, N_SIMULATION);
            action = random_action(state);
            state = state.next(action);
        }
    }
}

int main(void)
{
    time_t time = system_clock::to_time_t(system_clock::now());
    cout << "Starting program at " << ctime(&time) << endl;
    cout << "Parameter, CUDA_PLAYOUT, n_playout, N_SIMULATION, EXPAND_BASE" << endl << endl;

    extern int n_playout;
    if (MESURING_STRENGTH)
    {
        for (n_playout = 128; n_playout < 129; n_playout *= 2)
        {
            int win = 0, lose = 0, draw = 0;
            const int play_num = 150;
    
            printf("Parameter, %d, %d, %d, %d\n", CUDA_PLAYOUT, n_playout, N_SIMULATION, EXPAND_BASE);

            for (int i = 0; i < play_num; i++)
            {
                cout << "play, " << i << endl;
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
                cout << "black, " << win << endl;
                cout << "white, " << lose << endl;
                cout << "draw, " << draw << endl;
            }

            cout << endl;
        }
    }
    else
    {
        for (n_playout = 8192; n_playout <= 65536; n_playout += 8704){
            int win = 0, lose = 0, draw = 0;
            const int play_num = 5;
    
            printf("Parameter, %d, %d, %d, %d\n", CUDA_PLAYOUT, n_playout, N_SIMULATION, EXPAND_BASE);

            for (int i = 0; i < play_num; i++)
            {
                cout << "play, " << i << endl;
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
        }
    }

    time = system_clock::to_time_t(system_clock::now());
    cout << "Program end at " << ctime(&time) << endl;
}
