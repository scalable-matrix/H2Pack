LIB     = ../src/libH2Pack.a #$(VTUNE_AMPLIFIER_2018_DIR)/lib64/libittnotify.a
INC     = -I../src/include   #-I$(VTUNE_AMPLIFIER_2018_DIR)/include
CC      = icc
DEF     = -DUSE_MKL -mkl -DUSE_AVX
CFLAGS  = $(INC) -Wall -g -std=gnu99 -O3 -qopenmp -xHost $(DEF)
LDFLAGS = -g -O3 -xHost -qopenmp $(DEF)

SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)
EXES = $(OBJS:.o=.exe)

all: $(EXES)

%.o: %.c
	$(CC) $(CFLAGS) -I../ -c $^ 
	
%.exe: %.o $(LIB)
	$(CC) $(LDFLAGS) -o $@ $^ 
	
clean:
	rm -f $(EXES) $(OBJS)
