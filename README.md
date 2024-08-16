# zig-cuda-utils
[WIP!] Library with some GPU accelerated math functions for zig

Currently just a small, easy to understand demo of how to compile cuda libraries and link them into zig


Current settings: 2048x2048 matmul
![image](https://github.com/user-attachments/assets/2df1c094-68f6-4b92-8de5-2669c3a400cb)



# instructions

change `N` in `src/main.zig`'s `main()` function to whatever size you want for your matrices

in `./cudalib` run

`nvcc -shared -o libmatmul.so -Xcompiler -fPIC matmul.cu`


then in `./` run
`zig build`

then run the executable in `./zig-out/bin/`


personally I just run `clear;nvcc -shared -o ./cudalib/libmatmul.so -Xcompiler -fPIC ./cudalib/matmul.cu;zig build
`
