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
            state.game.show_board();
            state.game.show_score();
            if (state.is_draw())
            {
                cout << "draw" << endl;
                return 0;
            }
            else if (state.is_lose())
            {
                cout << "lose" << endl;
                return -1;
            }
            else
            {
                cout << "win" << endl;
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
            int simulation = 100;
            Node root_node = Node(state, 20);
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

    cout << "CUDA_PLAYOUT: " << CUDA_PLAYOUT << ", N_PLAYOUT: " << N_PLAYOUT << endl;

    for (int i = 0; i < play_num; i++)
    {
        cout << "<play: " << i << ">" << endl;
        int result = play_othello();
        cout << endl;
        if (result == 0)
            draw++;
        else if (result == 1)
            win++;
        else
            lose++;
    }

    cout << "win: " << win << endl;
    cout << "lose: " << lose << endl;
    cout << "draw: " << draw << endl;
}
