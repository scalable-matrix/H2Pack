EXES  = test_SPDHSS_H2_scalar.exe
OBJS  = pcg.o block_jacobi_precond.o

USE_MKL      = 1
USE_OPENBLAS = 0
OPENBLAS_INSTALL_DIR = ../../OpenBLAS-0.3.8/install

CC      = icc
DEF     = 
INC     = -I../../src/include
LIB     = ../../src/libH2Pack.a
CFLAGS  = $(INC) -Wall -g -std=gnu99 -O3 $(DEF)
LDFLAGS = -g -O3

ifeq ($(CC), icc)
CFLAGS  += -qopenmp -xHost
endif

ifeq ($(CC), gcc)
CFLAGS  += -fopenmp -march=native -Wno-unused-result -Wno-unused-function
endif

ifeq ($(USE_MKL), 1)
DEF     += -DUSE_MKL
CFLAGS  += -mkl
LDFLAGS += -mkl -qopenmp
endif

ifeq ($(USE_OPENBLAS), 1)
DEF     += -DUSE_OPENBLAS
LIB     += $(OPENBLAS_INSTALL_DIR)/lib/libopenblas.a
INC     += -I$(OPENBLAS_INSTALL_DIR)/include
LDFLAGS += -fopenmp -lm
endif

all: $(EXES)

%.exe: %.o $(OBJS) $(LIB)
	$(CC) -o $@ $^ $(LDFLAGS) 

%.o: %.c
	$(CC) $(CFLAGS) -I../ -c $^ 

clean:
	rm -f $(EXES) $(OBJS)
