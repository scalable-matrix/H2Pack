H2PACK_INSTALL_DIR = ../../src

DEFS    = 
INCS    = -I$(H2PACK_INSTALL_DIR)
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O3 -fPIC $(DEFS)
LDFLAGS = -g -O3 -fopenmp
LIBS    = $(H2PACK_INSTALL_DIR)/libH2Pack.a

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
CFLAGS  += -fopenmp -xHost
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
CFLAGS  += -fopenmp -march=native -Wno-unused-result -Wno-unused-function
LIBS    += -lgfortran -lm
endif

ifeq ($(strip $(USE_MKL)), 1)
DEFS    += -DUSE_MKL
CFLAGS  += -mkl
LDFLAGS += -mkl
endif

ifeq ($(strip $(USE_OPENBLAS)), 1)
OPENBLAS_INSTALL_DIR = ../../OpenBLAS-git/install
DEFS    += -DUSE_OPENBLAS
INCS    += -I$(OPENBLAS_INSTALL_DIR)/include
LDFLAGS += -L$(OPENBLAS_INSTALL_DIR)/lib
LIBS    += -lopenblas
endif

C_SRCS 	= $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)
EXES    = example_regularHSS.exe example_SPDHSSH2.exe example_SPDHSSH2_tol.exe
SHARED_OBJS = pcg.c.o block_jacobi_precond.c.o LRD_precond.c.o FSAI_precond.c.o CSRPlus.c.o pcg_tests.c.o

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(C_OBJS)

all: $(EXES)

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

%.exe: %.c.o $(SHARED_OBJS) $(LIBS)
	$(CC) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(EXES) $(C_OBJS)
