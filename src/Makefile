LIB   = libH2Pack.a
OBJS  = H2Pack_typedef.o H2Pack_utils.o H2Pack_aux_structs.o H2Pack_ID_compress.o \
        H2Pack_partition.o H2Pack_build.o H2Pack_matvec.o

HOSTCC  = icc
AR      = xiar rcs
DEF     = -DUSE_MKL
INC     = ./include
CFLAGS  = -I$(INC) -Wall -g -std=gnu99 -O3 -qopenmp -xHost -mkl $(DEF)
LDFLAGS = -O3 -xHost -qopenmp 

$(LIB): $(OBJS) 
	${AR} $@ $^

H2Pack_typedef.o: $(INC)/*.h H2Pack_typedef.c
	$(HOSTCC) $(CFLAGS) H2Pack_typedef.c -c 

H2Pack_utils.o: $(INC)/*.h H2Pack_utils.c
	$(HOSTCC) $(CFLAGS) H2Pack_utils.c -c 

H2Pack_ID_compress.o: $(INC)/*.h H2Pack_ID_compress.c
	$(HOSTCC) $(CFLAGS) H2Pack_ID_compress.c -c 

H2Pack_aux_structs.o: $(INC)/*.h H2Pack_aux_structs.c
	$(HOSTCC) $(CFLAGS) H2Pack_aux_structs.c -c 

H2Pack_partition.o: $(INC)/*.h H2Pack_partition.c
	$(HOSTCC) $(CFLAGS) H2Pack_partition.c -c 

H2Pack_build.o: $(INC)/*.h H2Pack_build.c
	$(HOSTCC) $(CFLAGS) H2Pack_build.c -c 

H2Pack_matvec.o: $(INC)/*.h H2Pack_matvec.c
	$(HOSTCC) $(CFLAGS) H2Pack_matvec.c -c 

clean:
	rm $(OBJS) $(LIB)