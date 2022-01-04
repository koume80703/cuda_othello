#include "header/kernel.cuh"

__global__ void kernel(STATE_CUDA *sc, float *result, int seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /*
    sc[idx].turn = sc[0].turn;
    sc[idx].player = sc[0].player;
    sc[idx].winner = sc[0].winner;
    sc[idx].start_player = sc[0].start_player;
    sc[idx].was_passed = sc[0].was_passed;
    
    for (int y = 0; y < BOARD_SIZE+2; y++){
        for (int x = 0; x < BOARD_SIZE+2; x++){
            sc[idx].board[y][x] = sc[0].board[y][x];
        }
    }
    */

    STATE_CUDA sc_dst;
    sc_dst.turn = sc->turn;
    sc_dst.player = sc->player;
    sc_dst.winner = sc->winner;
    sc_dst.start_player = sc->start_player;
    sc_dst.was_passed = sc->was_passed;

    for (int y = 0; y < BOARD_SIZE+2; y++){
        for (int x = 0; x < BOARD_SIZE+2; x++){
            sc_dst.board[y][x] = sc->board[y][x];
        }
    }

    curandStateXORWOW_t s;
    curand_init(seed, idx, 0, &s);
    
    result[idx] = playout_gpu(&sc_dst, &s);
}

__device__ float playout_gpu(STATE_CUDA *sc, curandState* s){
    int la[LA_SIZE][2];
    
    while(1){
        if (is_done(sc)){
            if (is_win(sc)){
                return 1;
            } else if (is_lose(sc)){
                return -1;
            } else {
                return 0;
            }
        }

        for (int i = 0; i < LA_SIZE; i++){
            la[i][0] = -1;
            la[i][1] = -1;
        }
    
        legal_actions(sc, la);
    
        if (la[0][0] == -1){
            pass_moving(sc);
        } else {
            int random_index = random_action(sc, la, s);
            int x = la[random_index][0];
            int y = la[random_index][1];
            next(sc, x, y);
        }
    }
}

__device__ bool is_done(STATE_CUDA *sc){
    return sc->winner != -255;
}

__device__ bool is_draw(STATE_CUDA *sc){
    return sc->winner == 0; // 0 means "draw"
}

__device__ bool is_win(STATE_CUDA *sc){
    return !is_draw(sc) && sc->winner == START_PLAYER;
    // default: START_PLAYER = 1 (means black) 
}

__device__ bool is_lose(STATE_CUDA *sc){
    return !is_win(sc);
}

__device__ void legal_actions(STATE_CUDA *sc, int la[][2]){
    int la_index = 0;

    int flippable[FLIPPABLE_SIZE][2];
    for (int x = 1; x < BOARD_SIZE+1; x++){
        for (int y = 1; y < BOARD_SIZE+1; y++){
            if (sc->board[y][x] != 0)
                continue;
            for (int i = 0; i < FLIPPABLE_SIZE; i++){
                flippable[i][0] = -1;
                flippable[i][1] = -1;
            }
            flippable_stone(sc, x, y, flippable);
            if (flippable[0][0] == -1){
                continue;
            } else {
                la[la_index][0] = x;
                la[la_index][1] = y;
                la_index++;
            }
        }
    }
}

__device__ void flippable_stone(STATE_CUDA* sc, int x, int y, int flippable[][2]){
    int tmp[FLIPPABLE_SIZE][2];
    int flippable_index = 0;
    for (int dx = -1; dx <= 1; dx++){
        for (int dy = -1; dy <= 1; dy++){
            if (dx == 0 && dy == 0)
                continue;

            for (int i = 0; i < FLIPPABLE_SIZE; i++){
                tmp[i][0] = -1;
                tmp[i][1] = -1;
            }
            int depth = 0, tmp_index = 0;

            while (true){
                depth++;

                int rx = x + (dx * depth);
                int ry = y + (dy * depth);

                int board_type = sc->board[ry][rx];
                /* 
                   EMPTY = 0, WHITE = -1, BLACK = 1; WALL = 2, NONE = -255 
                */
                if (board_type == 2 || board_type == 0){
                    break;
                } else {
                    if (board_type == sc->player){
                        if (tmp[0][0] != -1){
                            for (int i = 0; tmp[i][0] != -1; i++){
                                flippable[flippable_index][0] = tmp[i][0];
                                flippable[flippable_index][1] = tmp[i][1];
                                flippable_index++;
                            }
                        } break;
                    } else {
                        tmp[tmp_index][0] = rx;
                        tmp[tmp_index][1] = ry;
                        tmp_index++;
                    }
                }
            }        
        }
    }
}
    

__device__ bool next(STATE_CUDA *sc, int x, int y){
    if (set_stone(sc, x, y)){
        sc->was_passed = false;
        if (sc->player == 1 || sc->player == -1){
            sc->player = (sc->player == 1) ? -1 : 1;
        } else {
            return false;
        }
        sc->turn++;
        return true;
    } else {
        return false;
    }
}

__device__ bool set_stone(STATE_CUDA *sc, int x, int y){
    if (sc->board[y][x] != 0)
        return false;
    
    int flippable[FLIPPABLE_SIZE][2];
    for (int i = 0; i < FLIPPABLE_SIZE; i++){
        flippable[i][0] = -1;
        flippable[i][1] = -1;
    }
    flippable_stone(sc, x, y, flippable);
    if (flippable[0][0] == -1)
        return false;

    sc->board[y][x] = sc->player;
    for (int i = 0; flippable[i][0] != -1; i++){
        int x1 = flippable[i][0];
        int y1 = flippable[i][1];

        sc->board[y1][x1] = (sc->board[y1][x1] == 1) ? -1 : 1;
    }

    return true;
}

__device__ void pass_moving(STATE_CUDA *sc){
    if (sc->was_passed) return finish_game(sc);

    sc->was_passed = true;
    sc->player = (sc->player == 1) ? -1 : 1;
}

__device__ void finish_game(STATE_CUDA *sc){
    int white = 0, black = 0;
    for (int y = 1; y < BOARD_SIZE+1; y++){
        for (int x = 1; x < BOARD_SIZE+1; x++){
            if (sc->board[y][x] == -1){
                white++;
            } else {
                black++;
            }
        }
    }
    
    if (white < black){
        sc->winner = 1;  // PLAYER::BLACK
    } else if (black < white){
        sc->winner = -1; // PLAYER::WHITE
    } else {
        sc->winner = 0;  // PLAYER::DRAW
    }
}

__device__ int random_action(STATE_CUDA *sc, int la[][2], curandState* s){
    int irand = curand(s);

    if (irand < 0){
        irand = -irand;
    }

    int length = 0;
    for (int i = 0; la[i][0] != -1; i++){
        length++;
    }    
    return irand % length;
}
