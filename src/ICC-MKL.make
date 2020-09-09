LIBA  = libH2Pack.a
LIBSO = libH2Pack.so
OBJS  = H2Pack_typedef.o utils.o H2Pack_aux_structs.o H2Pack_ID_compress.o      \
		H2Pack_partition.o H2Pack_gen_proxy_point.o H2Pack_build.o              \
		H2Pack_matvec.o H2Pack_matmul.o H2Pack_utils.o H2Pack_HSS_ULV.o         \
		DAG_task_queue.o H2Pack_SPDHSS_H2.o		                                \
		H2Pack_partition_periodic.o H2Pack_build_periodic.o 					\
		H2Pack_matvec_periodic.o H2Pack_matmul_periodic.o 

USE_MKL      = 1
USE_OPENBLAS = 0
OPENBLAS_INSTALL_DIR = /home/mkurisu/Workspace/OpenBLAS-0.3.10/install

CC      = icc
DEF     = 
INC     = ./include
CFLAGS  = -I$(INC) -Wall -g -std=gnu99 -O3 -fPIC $(DEF)

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
AR      = xiar rcs
CFLAGS += -qopenmp -xHost
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
AR      = ar rcs
CFLAGS += -fopenmp -lm -march=native -Wno-unused-result -Wno-unused-function
endif

ifeq ($(strip $(USE_MKL)), 1)
DEF    += -DUSE_MKL
CFLAGS += -mkl
endif

ifeq ($(strip $(USE_OPENBLAS)), 1)
DEF    += -DUSE_OPENBLAS
INC    += -I$(OPENBLAS_INSTALL_DIR)/include
endif

all: $(LIBA) $(LIBSO)

$(LIBA): $(OBJS) 
	${AR} $@ $^

$(LIBSO): $(OBJS) 
	${CC} -shared -o $@ $^

H2Pack_typedef.o: ./include/*.h H2Pack_typedef.c
	$(CC) $(CFLAGS) H2Pack_typedef.c -c 

utils.o: ./include/*.h utils.c
	$(CC) $(CFLAGS) utils.c -c 

H2Pack_ID_compress.o: ./include/*.h H2Pack_ID_compress.c
	$(CC) $(CFLAGS) H2Pack_ID_compress.c -c 

H2Pack_aux_structs.o: ./include/*.h H2Pack_aux_structs.c
	$(CC) $(CFLAGS) H2Pack_aux_structs.c -c 

H2Pack_partition.o: ./include/*.h H2Pack_partition.c
	$(CC) $(CFLAGS) H2Pack_partition.c -c 

H2Pack_partition_periodic.o: ./include/*.h H2Pack_partition_periodic.c
	$(CC) $(CFLAGS) H2Pack_partition_periodic.c -c 

H2Pack_gen_proxy_point.o: ./include/*.h H2Pack_gen_proxy_point.c
	$(CC) $(CFLAGS) H2Pack_gen_proxy_point.c -c 

H2Pack_build.o: ./include/*.h H2Pack_build.c
	$(CC) $(CFLAGS) H2Pack_build.c -c 

H2Pack_build_periodic.o: ./include/*.h H2Pack_build_periodic.c
	$(CC) $(CFLAGS) H2Pack_build_periodic.c -c 

H2Pack_matvec.o: ./include/*.h H2Pack_matvec.c
	$(CC) $(CFLAGS) H2Pack_matvec.c -c 

H2Pack_matvec_periodic.o: ./include/*.h H2Pack_matvec_periodic.c
	$(CC) $(CFLAGS) H2Pack_matvec_periodic.c -c 

H2Pack_matmul.o: ./include/*.h H2Pack_matmul.c
	$(CC) $(CFLAGS) H2Pack_matmul.c -c 

H2Pack_matmul_periodic.o: ./include/*.h H2Pack_matmul_periodic.c
	$(CC) $(CFLAGS) H2Pack_matmul_periodic.c -c 

H2Pack_HSS_ULV.o: ./include/*.h H2Pack_HSS_ULV.c
	$(CC) $(CFLAGS) H2Pack_HSS_ULV.c -c 

H2Pack_utils.o: ./include/*.h H2Pack_utils.c
	$(CC) $(CFLAGS) H2Pack_utils.c -c 

H2Pack_SPDHSS_H2.o: ./include/*.h H2Pack_SPDHSS_H2.c
	$(CC) $(CFLAGS) H2Pack_SPDHSS_H2.c -c 

DAG_task_queue.o: ./include/*.h DAG_task_queue.c
	$(CC) $(CFLAGS) DAG_task_queue.c -c 

clean:
	rm $(OBJS) $(LIB)
