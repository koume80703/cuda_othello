#pragma once

#include <iostream>
#include <vector>
#include <random>
#include "enum.hpp"
#include "game.hpp"

using namespace std;

class State
{
public:
    Game game;

    State();
    State(Game original);

    State(const State &rhs);
    State &operator=(const State &rhs);

    State next(pair<int, int> action);
    State pass_moving();
    vector<pair<int, int>> legal_actions();
    pair<int, int> random_action();
    PLAYER winner();
    bool is_win();
    bool is_lose();
    bool is_draw();
    bool is_done();
    bool is_first_player();
};