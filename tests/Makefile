LIB     = ../src/libH2Pack.a #$(VTUNE_AMPLIFIER_2018_DIR)/lib64/libittnotify.a
INC     = -I../src/include   #-I$(VTUNE_AMPLIFIER_2018_DIR)/include
HOSTCC  = icc
CFLAGS  = $(INC) -Wall -g -std=gnu99 -O3 -qopenmp -xHost -mkl -DUSE_AVX
LDFLAGS = -g -O3 -xHost -qopenmp -mkl 

SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)
EXES = $(OBJS:.o=.x)

all: $(EXES)

%.o: %.c
	$(HOSTCC) $(CFLAGS) -I../ -c $^ 
	
%.x: %.o $(LIB)
	$(HOSTCC) $(LDFLAGS) -o $@ $^ 
	
clean:
	rm -f $(EXES) $(OBJS)
