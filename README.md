# basic-cuda-utils
Library with some GPU accelerated math

Currently just a small, easy to understand demo of how to incorporate cuda into zig


# instructions

in `./cudalib` run

`nvcc -shared -o libmatmul.so -Xcompiler -fPIC matmul.cu`


then in `./` run
`zig build`

then run the executable output to `./zig-out/bin/`
