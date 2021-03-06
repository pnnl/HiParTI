Getting started with HiParTI
===========================


To use HiParTI in a C/C++ project, simply add
```c
#include <HiParTI.h>
```
to your source code.

Link your code with
```sh
-fopenmp -lHiParTI -lm
```

This intro document can help you get used to the basics of HiParTI.


Data types
----------

`ptiScalar`: the default real value data type. It is defined as `float` or `double` type. For some devices without 64-bit float point support or data is out-of-memory in `double` type, you might need to define `ptiScalar` as `float`.

`ptiVector`: dense dynamic array of `ptiScalar` type scalars. It is implemented as a one-dimensional array. It uses preallocation to reduce the overhead of the append operation.

`ptiSizeVector`: dense dynamic array of `size_t` type scalars. This is implemented twice to avoid templates for CUDA code.

`ptiMatrix`: dense matrix type. It is implemented as a two-dimensional array. Column count is aligned as multiples of 8.

`ptiSparseMatrix`: sparse matrix type in coordinate (COO) storage format. It stores the coordinates and the value of every non-zero entry.

`ptiSparseTensor`: sparse tensor type in coordinate (COO) storage format. It works similar to `ptiSparseMatrix`, but supports tensors with arbitrary modes (number of dimensions).

`ptiSemiSparseTensor`: semi-sparse tensor type in sCOO storage format (details explained in [our SC16-IA3 paper](http://fruitfly1026.github.io/static/files/sc16-ia3.pdf). It can be considered as "a sparse tensor with dense fibers".


Creating objects
----------------

Most data types can fit themselves into stack memory, as local variables. They will handle extra memory allocations on demand.

For example, to construct an `ptiVector` and use it.

```c
// Construct it
ptiVector my_vector;
ptiNewVector(&my_vector, 0, 0);

// Add values to it
ptiAppendVector(&my_vector, 42);
ptiAppendVector(&my_vector, 31);

// Copy it to another uninitialized vector
ptiVector another_vector;
ptiCopyVector(&another_vector, &my_vector);

// Access data
printf("%lf %lf\n", another_vector.data[0], another_vector.data[1]);

// Free memory
ptiFreeVector(&my_vector);
ptiFreeVector(&another_vector);
```

Most functions require initialized data structures. While functions named `New` or `Copy` require uninitialized data structions. They are states in the Doxygen document on a function basis. Failing to supply data with correct initialization state may result in memory leak or program crash.


Validation
----------

For the sake of simplicity, properties are not designed. You can directly modify any field of any struct.

Every function assumes the input is valid, and guarantees the output is valid. This reduces the the need to check the input for most of the time, and improves the performance as a math library.

But if you modify the data structure directly, you must keep it valid. Some functions expect ordered input, you should sort them with functions like `ptiSparseTensorSortIndex` after your modification, or the functions may not work correctly. These functions usually also produces ordered output.


Error reporting
---------------

Most functions return 0 when it succeeded, non-zero when failed.

By invoking `ptiGetLastError`, you can extract the last error information.

Operating system `errno` and CUDA error code are also captured and converted.

If you need to make sure a procedure produces no error, call `ptiClearLastError` first, since success procedures does not clear last error status automatically.

Limitation: Memory might not be released properly when an error happened. The application will be in an inconsistent state. This might be fixed in future releases.
