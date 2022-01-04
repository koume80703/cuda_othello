#pragma once

enum class BOARD_STATE :int{
    EMPTY = 0,
    WHITE = -1,
    BLACK = 1,
    WALL = 2,

    NONE = -255,
};

enum class PLAYER :int{
    WHITE = -1,
    BLACK = 1,
    DRAW = 0,

    NONE = -255,
};