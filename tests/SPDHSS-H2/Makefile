EXES  = test_SPDHSS_H2_scalar.exe test_SPDHSS_H2_tensor.exe
OBJS  = pcg.o block_jacobi_precond.o LRD_precond.o FSAI_precond.o CSRPlus.o pcg_tests.o

USE_MKL      = 1
USE_OPENBLAS = 0
OPENBLAS_INSTALL_DIR = ../../../OpenBLAS-0.3.8/install

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

test_SPDHSS_H2_scalar.exe: test_SPDHSS_H2_scalar.o $(OBJS) $(LIB)
	$(CC) -o $@ $^ $(LDFLAGS) 

test_SPDHSS_H2_tensor.exe: test_SPDHSS_H2_tensor.o $(OBJS) $(LIB)
	$(CC) -o $@ $^ $(LDFLAGS) 

pcg.o: pcg.c pcg.h
	$(CC) $(CFLAGS) -I../ -c pcg.c

block_jacobi_precond.o: block_jacobi_precond.c block_jacobi_precond.h
	$(CC) $(CFLAGS) -I../ -c block_jacobi_precond.c

LRD_precond.o: LRD_precond.c LRD_precond.h
	$(CC) $(CFLAGS) -I../ -c LRD_precond.c

FSAI_precond.o: FSAI_precond.c FSAI_precond.h CSRPlus.h
	$(CC) $(CFLAGS) -I../ -c FSAI_precond.c

CSRPlus.o: CSRPlus.c CSRPlus.h
	$(CC) $(CFLAGS) -I../ -c CSRPlus.c

pcg_tests.o: pcg_tests.c pcg_tests.h
	$(CC) $(CFLAGS) -I../ -c pcg_tests.c

test_SPDHSS_H2_scalar.o: test_SPDHSS_H2_scalar.c
	$(CC) $(CFLAGS) -I../ -c test_SPDHSS_H2_scalar.c

test_SPDHSS_H2_tensor.o: test_SPDHSS_H2_tensor.c
	$(CC) $(CFLAGS) -I../ -c test_SPDHSS_H2_tensor.c

clean:
	rm -f $(EXES) $(OBJS)
