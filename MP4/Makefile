CC = nvcc
CU = template
NAME = mp4

EXP = ./data/${i}/output.dat
IN = ./data/0/input.dat,./data/0/kernel.dat
OUT = .
TYPE = vector # vector, matrix, or image

CCFLAGS = -std=c++11 -rdc=true -c $(CU).cu  -o $(CU).o 
LDFLAGS = -std=c++11 -o $(NAME) $(CU).o ../libwb/lib/libwb.so

local:
	$(CC) $(CCFLAGS)
	$(CC) $(LDFLAGS)

run:
	./$(NAME) -e EXP -i IN -o OUT -t TYPE
	
