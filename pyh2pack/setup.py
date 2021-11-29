from distutils.core import setup, Extension
import os
import numpy

H2PACK_DIR = ".."
OPENBLAS_INSTALL_DIR = "/usr/local/opt/openblas"
#C_DIR = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include"

extra_cflags  = ["-I"+H2PACK_DIR+"/include"]
extra_cflags += ["-I"+OPENBLAS_INSTALL_DIR+"/include"]
extra_cflags += ["-g", "-std=gnu99", "-O3"]
extra_cflags += ["-DUSE_OPENBLAS", "-fopenmp", "-march=native"]
extra_cflags += ["-Wno-unused-result", "-Wno-unused-function"]

LIB = [H2PACK_DIR+"/lib/libH2Pack.a", OPENBLAS_INSTALL_DIR+"/lib/libopenblas.a"]
extra_lflags = LIB + ["-g", "-O3", "-fopenmp", "-lm", "-lgfortran"]

def main():
    setup(name="pyh2pack",
        version="1.0.0",
        description="Python interface for H2Pack",
        author="Hua Huang, Xin Xing, and Edmond Chow",
        author_email="xxing02@gmail.com",
        ext_modules=[Extension(
            name = "pyh2pack",
            sources = ["pyh2pack.c"],
            include_dirs=[H2PACK_DIR+"/include", numpy.get_include()],
            extra_compile_args = extra_cflags,
            extra_link_args= extra_lflags,
            )
        ]
    )

if __name__ == "__main__":
    main()
