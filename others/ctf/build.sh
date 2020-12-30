rm $CTF/bin/ctf
mpicxx -std=c++0x -std=c++0x -fopenmp -Wall  -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -DFTN_UNDERSCORE=1  ctf.cxx -o $CTF/bin/ctf -I$CTF_INSTALL/include/ -L$CTF_INSTALL/lib -lctf -lblas -g
