#include "header/trans_data.hpp"

void trans_data(State state, STATE_CUDA *sc)
{
    Game game = state.game;
    Board board = game.get_board();

    trans_board(board, sc->board);

    sc->turn = game.get_turn();
    sc->player = player_to_int(game.get_current_player());
    sc->winner = -255; // PLAYER::NONE
    sc->was_passed = game.get_was_passed();
    sc->start_player = START_PLAYER;

    // show_board(sc->board);

    // cout << sc->turn << ", " << sc->player << ", " << sc->winner << ", " << sc->was_passed << endl;
    return;
}

void trans_board(Board board, int bArray[][BOARD_SIZE + 2])
{
    for (int y = 0; y < BOARD_SIZE + 2; y++)
    {
        for (int x = 0; x < BOARD_SIZE + 2; x++)
        {
            bArray[y][x] = board_state_to_int(board.get_cell(x, y));
        }
    }
}

int player_to_int(PLAYER p)
{
    if (p == PLAYER::WHITE)
        return -1;
    if (p == PLAYER::BLACK)
        return 1;
    if (p == PLAYER::DRAW)
        return 0;
    if (p == PLAYER::NONE)
        return -255;

    cout << "error: invalid argument in <player_to_int>" << endl;
    return 255;
}

int board_state_to_int(BOARD_STATE bs)
{
    if (bs == BOARD_STATE::EMPTY)
        return 0;
    if (bs == BOARD_STATE::WHITE)
        return -1;
    if (bs == BOARD_STATE::BLACK)
        return 1;
    if (bs == BOARD_STATE::WALL)
        return 2;
    if (bs == BOARD_STATE::NONE)
        return -255;

    cout << "error: invalid argument in <board_state_to_int>" << endl;
    return 255;
}

void show_board(int board[][BOARD_SIZE + 2])
{
    for (int i = 0; i < 20; i++)
    {
        printf("--");
    }
    printf("\n");
    for (int i = 0; i < BOARD_SIZE + 2; i++)
    {
        for (int j = 0; j < BOARD_SIZE + 2; j++)
        {
            if (board[i][j] == -1)
                printf("w ");
            else if (board[i][j] == 1)
                printf("b ");
            else if (board[i][j] == 0)
                printf("* ");
            else
                printf(". ");
        }
        printf("\n");
    }
    for (int i = 0; i < 20; i++)
    {
        printf("--");
    }
    printf("\n");
}
