LIB     = libH2Pack.a
OBJS    = utils.o

HOSTCC  = icc
AR      = xiar rcs
DEF     = -DUSE_MKL
INC     = ./include
CFLAGS  = -I$(INC) -Wall -g -std=gnu99 -O3 -qopenmp -xHost -mkl $(DEF)
LDFLAGS = -O3 -xHost -qopenmp 

$(LIB): $(OBJS) 
	${AR} $@ $^
    
utils.o: $(INC)/config.h $(INC)/utils.h utils.c
	$(HOSTCC) $(CFLAGS) utils.c -c 
    
clean:
	rm $(OBJS) $(LIB)