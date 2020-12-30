1. Install CTF and set up the following paths:

CTF_INSTALL: CTF installation path
CTF: CTF source path
TENSORS: tensors path

For example:

export CTF_INSTALL=/home/jiawen/ctf/ctf_install

export CTF=/home/jiawen/ctf

export TENSORS=/home/jiawen/tensors

2. Either put the folder in $CTF or change line 17 - 20 in ctf.cxx with the correct path:

For example:

#include "/home/jiawen/ctf/src/shared/util.h"

#include "/home/jiawen/ctf/src/interface/world.h"

#include "/home/jiawen/ctf/src/interface/tensor.h"

#include "/home/jiawen/ctf/src/interface/back_comp.h"

3. In ctf.cxx, -a and -b refer to the index symbols of the input tensors A and B. -f refers to which index of free-mode in the input tensor.

For example: $CTF/bin/ctf -A $TENSORS/4d_3_16.tns -B $TENSORS/4d_3_16.tns -a 0123 -b 0124 -f 3.
