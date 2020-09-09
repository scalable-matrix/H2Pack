LIB     = ../src/libH2Pack.a 
INC     = -I../src/include   

USE_MKL      = 1
USE_OPENBLAS = 0
OPENBLAS_INSTALL_DIR = /home/mkurisu/Workspace/OpenBLAS-0.3.10/install

CC      = icc
DEF     = 
CFLAGS  = $(INC) -Wall -g -std=gnu99 -O3 $(DEF)
LDFLAGS = -g -O3

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
CFLAGS  += -qopenmp -xHost
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
CFLAGS  += -fopenmp -lm -march=native -Wno-unused-result -Wno-unused-function
LDFLAGS += -lgfortran
endif

ifeq ($(strip $(USE_MKL)), 1)
DEF     += -DUSE_MKL
CFLAGS  += -mkl
LDFLAGS += -mkl -qopenmp
endif

ifeq ($(strip $(USE_OPENBLAS)), 1)
DEF     += -DUSE_OPENBLAS
LIB     += $(OPENBLAS_INSTALL_DIR)/lib/libopenblas.a
INC     += -I$(OPENBLAS_INSTALL_DIR)/include
LDFLAGS += -fopenmp -lm
endif

SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)
EXES = $(OBJS:.o=.exe)

all: $(EXES)

%.o: %.c
	$(CC) $(CFLAGS) -I../ -c $^ 

%.exe: %.o $(LIB)
	$(CC) -o $@ $^ $(LDFLAGS) 

clean:
	rm -f $(EXES) $(OBJS)
