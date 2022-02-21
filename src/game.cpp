#include "header/game.hpp"

Game::Game() : START_PLAYER(PLAYER::BLACK)
{
    Board board;
    player = START_PLAYER;
    turn = 0;
    winner = PLAYER::NONE;
    was_passed = false;

    vector<int> stone_num;
}

Game::Game(const Game &rhs) : START_PLAYER(rhs.START_PLAYER)
{
    board = rhs.board;
    player = rhs.player;
    turn = rhs.turn;
    winner = rhs.winner;
    was_passed = rhs.was_passed;

    stone_num = rhs.stone_num;
}

Game &Game::operator=(const Game &rhs)
{
    board = rhs.board;

    player = rhs.player;
    turn = rhs.turn;
    winner = rhs.winner;
    was_passed = rhs.was_passed;

    stone_num = rhs.stone_num;

    return *this;
}

PLAYER Game::state_to_player(BOARD_STATE bs)
{
    if (bs == BOARD_STATE::WHITE)
    {
        return PLAYER::WHITE;
    }
    else if (bs == BOARD_STATE::BLACK)
    {
        return PLAYER::BLACK;
    }
    else
    {
        cout << "error: invalid value in <state_to_player>" << endl;
        return PLAYER::NONE;
    }
}

BOARD_STATE Game::player_to_state(PLAYER p)
{
    if (p == PLAYER::WHITE)
    {
        return BOARD_STATE::WHITE;
    }
    else if (p == PLAYER::BLACK)
    {
        return BOARD_STATE::BLACK;
    }
    else
    {
        cout << "error: invalid value in <state_to_player>" << endl;
        return BOARD_STATE::NONE;
    }
}

bool Game::is_finished()
{
    return winner != PLAYER::NONE;
}

vector<pair<int, int>> Game::vector_placable_stone()
{
    return board.vector_placable_stone(player_to_state(player));
}

string Game::get_color(PLAYER player)
{
    if (player == PLAYER::WHITE)
        return "WHITE";
    if (player == PLAYER::BLACK)
        return "BLACK";

    return "DRAW";
}

PLAYER Game::get_current_player()
{
    return player;
}

PLAYER Game::get_next_player()
{
    return (player == PLAYER::BLACK) ? PLAYER::WHITE : PLAYER::BLACK;
}

void Game::shift_player()
{
    player = get_next_player();
}

bool Game::set_stone(int x, int y)
{
    if (board.set_stone(x, y, player_to_state(player)))
    {
        was_passed = false;
        shift_player();
        turn++;
        return true;
    }
    else
    {
        return false;
    }
}

void Game::pass_moving()
{
    if (was_passed)
        return finish_game();

    was_passed = true;
    shift_player();
}

void Game::show_score()
{
    cout << "WHITE: " << stone_num[0] << endl;
    cout << "BLACK: " << stone_num[1] << endl;
}

vector<int> Game::get_stone_num()
{
    int white = 0;
    int black = 0;

    for (int x = 1; x < board.get_size() + 1; x++)
    {
        for (int y = 1; y < board.get_size() + 1; y++)
        {
            if (board.get_cell(x, y) == BOARD_STATE::WHITE)
            {
                white++;
            }
            else
            {
                black++;
            }
        }
    }

    vector<int> tmp{white, black};
    return tmp;
}

void Game::finish_game()
{
    stone_num = get_stone_num();
    int white = stone_num[0];
    int black = stone_num[1];

    if (white < black)
    {
        winner = PLAYER::BLACK;
    }
    else if (black < white)
    {
        winner = PLAYER::WHITE;
    }
    else
    {
        winner = PLAYER::DRAW;
    }
}

bool Game::is_win(PLAYER base_player)
{
    return !is_draw() &&  base_player == winner;
}

bool Game::is_lose(PLAYER base_player)
{
    return !is_draw() && !is_win(base_player);
}

bool Game::is_draw()
{
    return winner == PLAYER::DRAW;
}

bool Game::is_first_player()
{
    return START_PLAYER == player;
}

void Game::show_board()
{
    board.show_board();
}

PLAYER Game::get_winner()
{
    return winner;
}

Board &Game::get_board()
{
    return board;
}

int Game::get_turn()
{
    return turn;
}

bool Game::get_was_passed()
{
    return was_passed;
}
