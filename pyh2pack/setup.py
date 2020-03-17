from distutils.core import setup, Extension
import os
import numpy

H2PACK_DIR = os.path.dirname(os.getcwd()) + "/src"
OPENBLAS_INSTALL_DIR = "/usr/local/opt/openblas"

extra_cflags = ["-I"+H2PACK_DIR+"/include"]
extra_cflags += ["-I"+OPENBLAS_INSTALL_DIR+"/include"]
extra_cflags += ["-Wall", "-g", "-std=gnu99", "-O2", "-DUSE_AVX"]
extra_cflags += ["-DUSE_OPENBLAS", "-fopenmp", "-march=native"]
extra_cflags += ["-Wno-unused-result", "-Wno-unused-function"]

LIB = [H2PACK_DIR+"/libH2Pack.a", OPENBLAS_INSTALL_DIR+"/lib/libopenblas.a"]
extra_lflags = LIB + ["-g", "-O2", "-fopenmp", "-lm"]

def main():
    setup(name="pyh2pack",
        version="1.0.0",
        description="Python interface for H2Pack",
        author="Hua Huang, and Xin Xing",
        author_email="xxing02@gmail.com",
        ext_modules=[Extension(
            name = "pyh2pack",
            sources = ["pyh2pack.c"], 
            include_dirs=["/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include", "../src/include", numpy.get_include()],
            extra_compile_args = extra_cflags, 
            extra_link_args= extra_lflags, 
            )   
        ]
    )

if __name__ == "__main__":
    main()