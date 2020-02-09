LIB   = libH2Pack.a
OBJS  = H2Pack_typedef.o H2Pack_utils.o H2Pack_aux_structs.o H2Pack_ID_compress.o \
        H2Pack_partition.o H2Pack_build.o H2Pack_matvec.o

CC      = icc
AR      = xiar rcs
DEF     = -DUSE_MKL -mkl -DUSE_AVX
INC     = ./include
CFLAGS  = -I$(INC) -Wall -g -std=gnu99 -O3 -qopenmp -xHost $(DEF)
LDFLAGS = -O3 -xHost -qopenmp 

$(LIB): $(OBJS) 
	${AR} $@ $^

H2Pack_typedef.o: $(INC)/*.h H2Pack_typedef.c
	$(CC) $(CFLAGS) H2Pack_typedef.c -c 

H2Pack_utils.o: $(INC)/*.h H2Pack_utils.c
	$(CC) $(CFLAGS) H2Pack_utils.c -c 

H2Pack_ID_compress.o: $(INC)/*.h H2Pack_ID_compress.c
	$(CC) $(CFLAGS) H2Pack_ID_compress.c -c 

H2Pack_aux_structs.o: $(INC)/*.h H2Pack_aux_structs.c
	$(CC) $(CFLAGS) H2Pack_aux_structs.c -c 

H2Pack_partition.o: $(INC)/*.h H2Pack_partition.c
	$(CC) $(CFLAGS) H2Pack_partition.c -c 

H2Pack_build.o: $(INC)/*.h H2Pack_build.c
	$(CC) $(CFLAGS) H2Pack_build.c -c 

H2Pack_matvec.o: $(INC)/*.h H2Pack_matvec.c
	$(CC) $(CFLAGS) H2Pack_matvec.c -c 

clean:
	rm $(OBJS) $(LIB)