CC           = gcc
USE_MKL      = 0
USE_OPENBLAS = 1

include common.make

# GCC 10 need to manually specify using SVE, -march=native is not enough
# On A64FX SVE vector bits = 512, on other SVE supported processors this value might be different
USE_AARCH64_SVE = 0
SVE_VECTOR_BITS = 512
ifeq ($(strip $(USE_AARCH64_SVE)), 1)
CFLAGS := $(subst -march=native, -march=armv8.2-a+sve -msve-vector-bits=$(SVE_VECTOR_BITS), $(CFLAGS))
endif