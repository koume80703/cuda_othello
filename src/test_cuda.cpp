#include "header/trans_data.hpp"
#include "header/playout_cuda.cuh"

int main(void){
    Game game;
    State state = State(game);

    playout_cuda(state);
}
