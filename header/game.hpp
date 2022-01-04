#pragma once

#include <iostream>
#include <vector>
#include "enum.hpp"
#include "board.hpp"

using namespace std;

class Game{
private:
    int turn;
    Board board;
        
    PLAYER player, winner;
    const PLAYER START_PLAYER;
        
    bool was_passed;
        
    vector<int> stone_num;

public:
    Game();

    Game(const Game& rhs);
    Game& operator=(const Game& rhs);

    PLAYER state_to_player(BOARD_STATE bs);
    BOARD_STATE player_to_state(PLAYER p);
        
    bool is_finished();
    vector< pair<int, int> > vector_placable_stone();
    string get_color(PLAYER player);
    PLAYER get_current_player();
    PLAYER get_next_player();
    void shift_player();
    bool set_stone(int x, int y);
    void pass_moving();
    void show_score();
    vector<int> get_stone_num();
    void finish_game();

    bool is_win();
    bool is_lose();
    bool is_draw();
    bool is_first_player();
    void show_board();

    PLAYER get_winner();
    Board& get_board();
    int get_turn();
    bool get_was_passed();
};
