#include "header/state.hpp"

State::State()
{
}

State::State(Game original) : game(original)
{
}

State::State(const State &rhs)
{
    game = rhs.game;
}

State &State::operator=(const State &rhs)
{
    game = rhs.game;
    return *this;
}

State State::next(pair<int, int> action)
{
    State n_state = State(game);
    n_state.game.set_stone(action.first, action.second);
    return n_state;
}

State State::pass_moving()
{
    State n_state = State(game);
    n_state.game.pass_moving();
    return n_state;
}

vector<pair<int, int>> State::legal_actions()
{
    return game.vector_placable_stone();
}

pair<int, int> State::random_action()
{
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> rid(0, legal_actions().size() - 1);
    int irand = rid(mt);
    return legal_actions()[irand];
}

PLAYER State::winner()
{
    return game.get_winner();
}

bool State::is_win()
{
    return game.is_win();
}

bool State::is_lose()
{
    return game.is_lose();
}

bool State::is_draw()
{
    return game.is_draw();
}

bool State::is_done()
{
    return game.is_finished();
}

bool State::is_first_player()
{
    return game.is_first_player();
}
