#pragma once

#include <vector>
#include <iostream>

#include "state.hpp"
#include "game.hpp"
#include "board.hpp"
#include "enum.hpp"

#define BOARD_SIZE 8
#define START_PLAYER 1

typedef struct _STATE_CUDA
{
  int board[BOARD_SIZE + 2][BOARD_SIZE + 2];
  /*
  EMPTY = 0, WHITE = -1, BLACK = 1, WALL = 2, NONE = -255
  */

  int turn;
  int player, winner, base_player;
  /*
  WHITE = -1, BLACK = 1, DRAW = 0, NONE = -255
  */

  bool was_passed;
} STATE_CUDA;

void trans_data(State state, PLAYER base_player, STATE_CUDA *dfc);
void trans_board(Board board, int bArray[][BOARD_SIZE + 2]);
int player_to_int(PLAYER p);
int board_state_to_int(BOARD_STATE bs);
void show_board(int board[][BOARD_SIZE + 2]);
