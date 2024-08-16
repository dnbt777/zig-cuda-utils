# basic-cuda-utils
Library with some GPU accelerated math

Currently just a small, easy to understand demo of how to incorporate cuda into zig


# instructions

in `./cudalib` run

`nvcc -shared -o libmatmul.so -Xcompiler -fPIC matmul.cu`


then in `./` run
`zig build`

then run the executable output to `./zig-out/bin/`


![image](https://github.com/user-attachments/assets/b3376d03-552f-48f1-83bf-8b0289f2db8d)
