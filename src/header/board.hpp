#pragma once

using namespace std;

#include <iostream>
#include <vector>
#include "enum.hpp"

class Board{
public:
    Board();

    Board(const Board& rhs);
    Board& operator=(const Board& rhs);

    bool set_stone(int x, int y, BOARD_STATE player);
    vector< pair<int, int> > vector_flippable_stone(int x, int y, BOARD_STATE player);
    vector< pair<int, int> > vector_placable_stone(BOARD_STATE player);
    void show_board();
    vector< vector<BOARD_STATE> > get_board_state();
    
    BOARD_STATE get_cell(int x, int y);
    int get_size();
        
private:        
    vector< vector<BOARD_STATE> > RawBoard;
    const int BOARD_SIZE;
};
