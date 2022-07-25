## Building and Installing PyH2Pack

### Intel compiler (ICC) + Intel MKL
Use this command to compile:
```shell
LDSHARED="icc -shared" CC=icc python3 setup_icc.py install
```
Before running the python code, you need to manually preload the following MKL file: 
```shell
# Check if $MKLROOT is set correctly
# ls $MKLROOT/lib/intel64/libmkl_rt.so
export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_rt.so
```

### GNU compiler (GCC) + OpenBLAS

Install or compile OpenBLAS first, then modify `setup.py` and update variable `OPENBLAS_INSTALL_DIR` according to the location OpenBLAS is installed. Use this command to compile:

```shell
CC=gcc python3 setup.py install
```

If you see an error message like:

```text
copying build/lib.linux-x86_64-3.8/pyh2pack.cpython-38-x86_64-linux-gnu.so -> /usr/local/lib/python3.8/dist-packages
error: could not delete '/usr/local/lib/python3.8/dist-packages/pyh2pack.cpython-38-x86_64-linux-gnu.so': Permission denied
```

Then manually run:

```shell
sudo cp build/lib.linux-x86_64-3.8/pyh2pack.cpython-38-x86_64-linux-gnu.so /usr/local/lib/python3.8/dist-packages
```



## Using PyH2Pack

See `example.py`. 

If you want to try the data-driven sample point method instead of the default proxy point / proxy surface method, see `example_samplept.py`. 
