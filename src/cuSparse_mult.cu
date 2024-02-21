#include <stdio.h>
#include <iostream>
#include <cusparse_v2.h>
#include <cuda.h>
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include "cuSparse_mult.cuh"

// error check macros
#define CUSPARSE_CHECK(x) {cusparseStatus_t _c=x; if (_c != CUSPARSE_STATUS_SUCCESS) {printf("cusparse fail: %d, line: %d\n", (int)_c, __LINE__); exit(-1);}}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

inline void CudaSparseCheckCore(cusparseStatus_t code, const char *file, int line) {
   if (code != CUSPARSE_STATUS_SUCCESS) {
      fprintf(stderr,"Cuda Error %d : %s %s %d\n", code, cusparseGetErrorString(code), file, line);
      exit(code);
   }
}

#define CudaSparseCheck( test ) { CudaSparseCheckCore((test), __FILE__, __LINE__); }

// perform sparse-matrix multiplication C=AxB
void cusparse_multiply(cusp::csr_matrix<int, float,cusp::device_memory> &A,
		cusp::csr_matrix<int, float,cusp::device_memory> &B, cusp::csr_matrix<int, float,cusp::device_memory> &res) {

	/* Get internal array pointers */
	int *A_row_offsets = thrust::raw_pointer_cast(A.row_offsets.data());
	int *B_row_offsets = thrust::raw_pointer_cast(B.row_offsets.data());
	int *A_column_indices = thrust::raw_pointer_cast(A.column_indices.data());
	int *B_column_indices = thrust::raw_pointer_cast(B.column_indices.data());
	float *A_values = thrust::raw_pointer_cast(A.values.data());
	float *B_values = thrust::raw_pointer_cast(B.values.data());

	/* Declare arrays for output matrix */
	int *C_row_offsets, *C_column_indices;
	float *C_values;


	/* Creates handles for matrices */
	cusparseMatDescr_t descrA, descrB, descrC;
	cusparseStatus_t stat;
	stat = cusparseCreateMatDescr(&descrA);
	CUSPARSE_CHECK(stat);
	stat = cusparseCreateMatDescr(&descrB);
	CUSPARSE_CHECK(stat);
	stat = cusparseCreateMatDescr(&descrC);
	CUSPARSE_CHECK(stat);
	stat = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	CUSPARSE_CHECK(stat);
	stat = cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
	CUSPARSE_CHECK(stat);
	stat = cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
	CUSPARSE_CHECK(stat);
	stat = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	CUSPARSE_CHECK(stat);
	stat = cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
	CUSPARSE_CHECK(stat);
	stat = cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);

	/* NO COMPILA, PERO COMPILABA CON LA CLASE ANTERIOR, LAS LLAMADAS A CUSPARSE ESTAN BIEN */
	int m = A.num_rows;
	int n = B.num_cols;
	int k = B.num_rows;
	cusparseHandle_t handle;
	CudaSparseCheck(cusparseCreate(&handle));

	// assume matrices A, B and D are ready.
	int baseC, nnzC;
	csrgemm2Info_t info = nullptr;
	size_t bufferSize;
	void *buffer = nullptr;
	// nnzTotalDevHostPtr points to host memory
	int *nnzTotalDevHostPtr = &nnzC;
	float alpha = 1.0;
	CudaSparseCheck(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

	// step 1: create an opaque structure
	CudaSparseCheck(cusparseCreateCsrgemm2Info(&info));

	// step 2: allocate buffer for csrgemm2Nnz and csrgemm2
	cusparseMatDescr_t descr_D; // not used, created only for the sake of the arguments
	CudaSparseCheck(cusparseCreateMatDescr(&descr_D));
	cusparseScsrgemm2_bufferSizeExt(handle, m, n, k, &alpha, descrA, A.num_entries,
			A_row_offsets, A_column_indices, descrB, B.num_entries, B_row_offsets, B_column_indices, nullptr,
			descr_D, 0, nullptr, nullptr, info, &bufferSize);
	cudaMalloc(&buffer, bufferSize);

	// step 3: compute C.row_ptr
	cudaMalloc((void**) &C_row_offsets, sizeof(int) * (m + 1));
	cusparseXcsrgemm2Nnz(handle, m, n, k, descrA, A.num_entries, A_row_offsets, A_column_indices,
			descrB, B.num_entries, B_row_offsets, B_column_indices,
			descr_D, 0, nullptr, nullptr, descrC, C_row_offsets, nnzTotalDevHostPtr,
			info, buffer);
	if (nullptr != nnzTotalDevHostPtr) {
		nnzC = *nnzTotalDevHostPtr;
	} else {
		cudaMemcpy(&nnzC, C_row_offsets + m, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&baseC, C_row_offsets, sizeof(int), cudaMemcpyDeviceToHost);
		nnzC -= baseC;
	}

	// step 4: finish sparsity pattern and value of C
	cudaMalloc((void**) &C_column_indices, sizeof(int) * nnzC);
	cudaMalloc((void**) &C_values, sizeof(float) * nnzC);
	// Remark: set C.val to nullptr if only sparsity pattern is required.
	cusparseScsrgemm2(handle, m, n, k,
			&alpha, descrA, A.num_entries, A_values, A_row_offsets,
			A_column_indices, descrB, B.num_entries, B_values, B_row_offsets, B_column_indices,
			nullptr, descr_D, 0, nullptr, nullptr, nullptr,
			descrC, C_values, C_row_offsets, C_column_indices,
			info, buffer);
	cudaDeviceSynchronize();

/* Solo para comparar */
	res.num_rows = m;
	res.num_cols = n;
	res.num_entries = nnzC;
	res.column_indices = cusp::array1d<int, cusp::device_memory>(C_column_indices, C_column_indices + nnzC);
	res.row_offsets = cusp::array1d<int, cusp::device_memory>(C_row_offsets, C_row_offsets + m + 1);
	res.values = cusp::array1d<float, cusp::device_memory>(C_values, C_values + nnzC);

	// step 5: destroy the opaque structure
	cusparseDestroyCsrgemm2Info(info);
}
