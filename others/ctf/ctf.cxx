/** Copyright (c) 2011, Edgar Solomonik, all rights reserved.
  * \addtogroup benchmarks
  * @{
  * \addtogroup bench_contractions
  * @{
  * \brief Benchmarks arbitrary NS contraction
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <vector>
#include "../src/shared/util.h"
#include "../src/interface/world.h"
#include "../src/interface/tensor.h"
#include "../src/interface/back_comp.h"

using namespace CTF;

CTF_Tensor *make_prod(
    bool is_sparse,
    CTF_World &dw,
    std::string const & name,
    std::vector<int>  ndimsA,
    std::vector<int> &ndimsB,
    int  mode
) {
    int nmodes = ndimsA.size();
    ndimsA[mode] = ndimsB[1];
    std::vector<int> ns(nmodes, NS);
    CTF_Tensor *result = new CTF_Tensor(nmodes, is_sparse, ndimsA.data(), ns.data(), dw, Ring<double>(), name.c_str(), true);
    return result;
}

CTF_Tensor *make_B(
    CTF_World &dw,
    std::string const & name,
    std::vector<int> const &ndimsA,
    std::vector<int> &ndimsB,
    int mode,
    int sizeB
) {
    ndimsB.resize(2);
    ndimsB[0] = ndimsA[mode];
    ndimsB[1] = sizeB;
    std::vector<int> ns(2, NS);
    CTF_Tensor *result = new CTF_Tensor(2, false, ndimsB.data(), ns.data(), dw, Ring<double>(), name.c_str(), true);
    return result;
}

CTF_Tensor *read_tensor(
    std::string const & filename,
    bool is_sparse,
    CTF_World &dw,
    std::string const & name,
    std::vector<int> &ndims
) {
    FILE *f = fopen(filename.c_str(), "r");
    if(!f) {
        fprintf(stderr, "unable to open %s\n", filename.c_str());
        exit(2);
    }
    printf("Reading from %s.\n", filename.c_str());
    int nmodes;
    fscanf(f, "%d", &nmodes);
    ndims.resize(nmodes);
    for(int i = 0; i < nmodes; ++i) {
        fscanf(f, "%d", &ndims[i]);
    }
    std::vector<int> ns(nmodes, NS);
    CTF_Tensor *result = new CTF_Tensor(nmodes, is_sparse, ndims.data(), ns.data(), dw, Ring<double>(), name.c_str(), true);
    std::vector<int64_t> inds;
    std::vector<double> values;
    for(;;) {
        long ii, jj, kk, ll;
        assert(nmodes == 3 || nmodes == 4);
        if (nmodes == 3) {
          if(fscanf(f, "%ld%ld%ld", &ii, &jj, &kk) == 3) {
              ii--; jj--; kk--; // offset 1
              int64_t global_idx = ii + jj*ndims[0] + kk*ndims[0]*ndims[1];
              inds.push_back(global_idx);
          } else {
              goto read_done;
          }
        } else if (nmodes ==4) {
          if(fscanf(f, "%ld%ld%ld%ld", &ii, &jj, &kk, &ll) == 4) {
              ii--; jj--; kk--; ll--; // offset 1
              int64_t global_idx = ii + jj*ndims[0] + kk*ndims[0]*ndims[1] + ll*ndims[0] * ndims[1] * ndims[2];
              inds.push_back(global_idx);
          } else {
              goto read_done;
          }
        }
        double v;
        if(fscanf(f, "%lf", &v) == 1) {
            values.push_back(v);
        } else {
            goto read_done;
        }
    }
read_done:
    result->write(values.size(), inds.data(), values.data());
    fclose(f);
    printf("Read from %s, %ld records.\n", filename.c_str(), (long) values.size());
    return result;
}

int bench_contraction(
                      std::string const & filenamea,
                      std::string const & filenameb,
					  char const * iA,
					  char const * iB,
					  char * idx_f,
                      CTF_World   &dw
                 ){

  int rank, i, num_pes;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  int order_A, order_B, order_C, num_f;
  order_A = strlen(iA);
  order_B = strlen(iB);
  num_f = strlen(idx_f);
  char* iC = (char*) malloc(num_f * 2 * sizeof(char));
  order_C = strlen(iC);

  int NS_A[order_A];
  int NS_B[order_B];
  int NS_C[order_C];
  int n_A[order_A];
  int n_B[order_B];
  int n_C[order_C];

  for (i=0; i<order_A; i++){
    n_A[i] = 1;
    NS_A[i] = NS;
  }
  for (i=0; i<order_B; i++){
    n_B[i] = 1;
    NS_B[i] = NS;
  }
  for (i=0; i<order_C; i++){
    n_C[i] = 1;
    NS_C[i] = NS;
  }

  static std::vector<int> ndimsA, ndimsB, ndimsC;
  CTF_Tensor* A = read_tensor(filenamea, true, dw, "A", ndimsA);
  CTF_Tensor* B = read_tensor(filenamea, true, dw, "B", ndimsB);
  ndimsC.resize(order_C);

  for(int m = 0; m < num_f; ++m) {
    ndimsC[m] = ndimsA[idx_f[m]-48];
	ndimsC[num_f+m] = ndimsB[idx_f[m]-48];
	iC[m] = iA[idx_f[m]-48];
	iC[num_f+m] = iB[idx_f[m]-48];
  }

  std::vector<int> ns(order_C, NS);
  CTF_Tensor *C = new CTF_Tensor(order_C, 1, ndimsC.data(), ns.data(), dw, Ring<double>(), "C", true);

  /// Main computation
  double st_time = MPI_Wtime();
  (*C)[iC] = (*A)[iA]*(*B)[iB];
  double end_time = MPI_Wtime();

  printf("Time: %lf sec\n", end_time-st_time);

  return 1;
}

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int rank, np;
  int const in_num = argc;
  char ** input_str = argv;
  char const * A;
  char const * B;
  char * iA;
  char * iB;
  char * idx_f;
  int cmodes;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-A")){
    A = getCmdOption(input_str, input_str+in_num, "-A");
  } else A = "a.tns";
  if (getCmdOption(input_str, input_str+in_num, "-B")){
    B = getCmdOption(input_str, input_str+in_num, "-B");
  } else B = "b.tns";
  if (getCmdOption(input_str, input_str+in_num, "-a")){
    iA = getCmdOption(input_str, input_str+in_num, "-a");
  } else iA = "012";
  if (getCmdOption(input_str, input_str+in_num, "-b")){
    iB = getCmdOption(input_str, input_str+in_num, "-b");
  } else iB = "312";
  // The index of free modes
  if (getCmdOption(input_str, input_str+in_num, "-f")){
    idx_f = getCmdOption(input_str, input_str+in_num, "-f");
  } else idx_f = "0";

  CTF_World dw(argc, argv);
  bench_contraction(A, B, iA, iB, idx_f, dw);

  MPI_Finalize();
  return 0;
}
