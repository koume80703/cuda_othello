#include "header/board.hpp"
#include "header/toString.hpp"

Board::Board(): BOARD_SIZE(8){
    RawBoard = vector< vector<BOARD_STATE> >(BOARD_SIZE+2, vector<BOARD_STATE>(BOARD_SIZE+2));
    for (int i = 0; i < BOARD_SIZE+2; i++){
        for (int j = 0; j < BOARD_SIZE+2; j++){                    
            RawBoard[i][j] = BOARD_STATE::WALL;
        }
    }
    for (int i = 1; i < BOARD_SIZE+1; i++){
        for (int j = 1; j < BOARD_SIZE+1; j++){
            RawBoard[i][j] = BOARD_STATE::EMPTY;
        }
    }
    RawBoard[4][4] = BOARD_STATE::WHITE;
    RawBoard[5][5] = BOARD_STATE::WHITE;
    RawBoard[4][5] = BOARD_STATE::BLACK;
    RawBoard[5][4] = BOARD_STATE::BLACK;
}

Board::Board(const Board& rhs) : BOARD_SIZE(rhs.BOARD_SIZE){
    RawBoard = vector< vector<BOARD_STATE> >(BOARD_SIZE+2, vector<BOARD_STATE>(BOARD_SIZE+2));
    for (int i = 0; i < BOARD_SIZE+2; i++){
        for (int j = 0; j < BOARD_SIZE+2; j++){
            RawBoard[i][j] = rhs.RawBoard[i][j];
        }
    }
}

Board& Board::operator=(const Board& rhs){
    for (int i = 0; i < BOARD_SIZE+2; i++){
        for (int j = 0; j < BOARD_SIZE+2; j++){
            RawBoard[i][j] = rhs.RawBoard[i][j];
        }
    }
    return *this;
}

bool Board::set_stone(int x, int y, BOARD_STATE player){
    if (RawBoard[y][x] != BOARD_STATE::EMPTY)
        return false;
                
    vector< pair<int, int> > flippable = vector_flippable_stone(x,y,player);            
    if (flippable.empty()) 
        return false;

    RawBoard[y][x] = player;
    for (const auto& cell : flippable){                
        int x1 = cell.first;
        int y1 = cell.second;

        RawBoard[y1][x1] = (RawBoard[y1][x1] == BOARD_STATE::BLACK) ? BOARD_STATE::WHITE : BOARD_STATE::BLACK;
    }

    return true;
}

vector< pair<int, int> > Board::vector_flippable_stone(int x, int y, BOARD_STATE player){
    vector< pair<int, int> > flippable;

    for (int dx = -1; dx <= 1; dx++){
        for (int dy = -1; dy <= 1; dy++){                    
            if (dx == 0 && dy == 0) 
                continue;

            vector< pair<int,int> > tmp;
            int depth = 0;
            
            while(true){
                depth++;

                int rx = x + (dx * depth);
                int ry = y + (dy * depth);                        

                BOARD_STATE board_type = RawBoard[ry][rx];

                if (board_type == BOARD_STATE::WALL || board_type == BOARD_STATE::EMPTY){
                    break;
                } else {
                    if (board_type == player){
                        if (!tmp.empty()){
                            for (const auto& t : tmp){
                                flippable.push_back(t);
                            }
                        }
                        break;
                    } else {                                    
                        tmp.push_back(make_pair(rx,ry));
                    } 
                }
            }
        }
    }

    return flippable;
}

vector< pair<int, int> > Board::vector_placable_stone(BOARD_STATE player){
    vector< pair<int, int> > placable;

    for (int x = 1; x < BOARD_SIZE+1; x++){
        for (int y = 1; y < BOARD_SIZE+1; y++){
            if (RawBoard[y][x] != BOARD_STATE::EMPTY)
                continue;
            if (vector_flippable_stone(x, y, player).empty())
                continue;
            else
                placable.push_back(make_pair(x,y));
        }
    }

    return placable;
}

void Board::show_board(){
    for(int i = 0; i < 20; i++)
        printf("--");
    printf("\n");
    for(int i = 0; i < BOARD_SIZE+2; i++){
        for(int j = 0; j < BOARD_SIZE+2; j++){
            if(RawBoard[i][j] == BOARD_STATE::WHITE) 
                printf("w ");
            else if (RawBoard[i][j] == BOARD_STATE::BLACK) 
                printf("b ");
            else if (RawBoard[i][j] == BOARD_STATE::EMPTY) 
                printf("* ");
            else 
                printf(". ");
        }
        printf("\n");
    }
    for(int i = 0; i < 20; i++)
        printf("--");
    printf("\n");
}

vector< vector<BOARD_STATE> > Board::get_board_state(){
    return RawBoard;
}

BOARD_STATE Board::get_cell(int x, int y){
    return RawBoard[y][x];
}

int Board::get_size(){
    return BOARD_SIZE;
}
