CC = nvcc
PROGRAM = othello
SOURCE = main.o game.o state.o board.o node.o mcts.o argmax.o ucb1.o trans_data.o playout_cuda.o kernel.o measuring.o
CFLAGS= -c


all : $(SOURCE)
	$(CC) -o $(PROGRAM) $^

debug : $(SOURCE)
	$(CC) -g -o $(PROGRAM) $^

main.o : main.cpp
	$(CC) $(CFLAGS) $^

board.o : board.cpp 
	$(CC) $(CFLAGS) $^

game.o : game.cpp
	$(CC) $(CFLAGS) $^

state.o : state.cpp
	$(CC) $(CFLAGS) $^

node.o : node.cpp
	$(CC) $(CFLAGS) $^

mcts.o : mcts.cpp
	$(CC) $(CFLAGS) $^

argmax.o : util/argmax.cpp
	$(CC) $(CFLAGS) $^

ucb1.o : util/ucb1.cpp
	$(CC) $(CFLAGS) $^

trans_data.o : trans_data.cpp
	$(CC) $(CFLAGS) $^

playout_cuda.o : playout_cuda.cu
	$(CC) $(CFLAGS) $^

kernel.o : kernel.cu
	$(CC) $(CFLAGS) -lcurand $^

measuring.o : measuring.cpp
	$(CC) $(CFLAGS) $^

clean:; rm -f *.o 
