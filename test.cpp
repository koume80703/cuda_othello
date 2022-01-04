#include "header/trans_data.hpp"


int main(void){
    Game game;
    State state = State(game);


    STATE_CUDA* sc = (STATE_CUDA*)malloc(sizeof(STATE_CUDA));
    trans_data(state, sc);
    free(sc);
}
