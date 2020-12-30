Contributing to HiParTI
======================

This is the guide about contributing to HiParTI.


Language standard
-----------------

HiParTI mainly follows C99 standard, and must be compatible with GCC >=4.9.
CUDA code follows C++03 standard, which is the default of NVCC compiler.


Indentation and format
----------------------

Feel free to use any indent style, but please respect the original style of an existing file.


Naming convention
-----------------

C does not have namespace, thus it is important to keep names from conflicting. All HiParTI functions have names starting with `pti`. Private funcions start with `pti_`.

Names of functions and types follow `PascalCase`. Constants and enumerations follow `UPPER_CASE`. While variables are not restricted to a naming convention.


Error checking
--------------

`pti_CheckError`, `pti_CheckOSError`, `pti_CheckCudaError` are used to check for invalid input or environmental exceptions.

Use `assert` to check for some conditions that should never happen on a production system, such as wrong data produced by other parts of HiParTI. I/O error or invalid data from the outside should not go into this category.


Using `const`
-------------

`const` provides immutability check, optimizes code, and improves documentation clarity. Correct usage of `const` against pointers and arrays are required.


Licensing and copyright
-----------------------

Contribution to HiParTI must license the code under BSD 3-Clause. Put a copyright notice alongside with your name at the top of each file you modify.
