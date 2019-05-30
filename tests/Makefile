LIB     = ../src/libH2Pack.a
INC     = -I../src/include
HOSTCC  = icc
CFLAGS  = $(INC) -Wall -g -std=gnu99 -O3 -qopenmp -xHost -mkl
LDFLAGS = -O3 -xHost -qopenmp 

SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)
EXES = $(OBJS:.o=.x)

all: $(EXES)

%.o: %.c
	$(HOSTCC) $(CFLAGS) -I../ -c $^ 
	
%.x: %.o
	$(HOSTCC) $(CFLAGS) -o $@ $^ $(LIB)
	
clean:
	rm -f $(EXES) $(OBJS)
