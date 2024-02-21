#include <stdio.h>
#include <iostream>
#include <cusparse_v2.h>
#include <cuda.h>
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <chrono>
#include "cuSparse_spmv.cuh"
#include <iostream>
#include <fstream>

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
void cusparse_spmv(cusp::csr_matrix<int, float,cusp::device_memory> &A,
		float* v, float* u, cusparseHandle_t handle, std::ofstream& file) {
        std::chrono::steady_clock::time_point start,end;
	

	/* Get internal array pointers */
	int *A_row_offsets = thrust::raw_pointer_cast(A.row_offsets.data());
	int *A_column_indices = thrust::raw_pointer_cast(A.column_indices.data());
	float *A_values = thrust::raw_pointer_cast(A.values.data());


	/* Creates handles for matrix */
	/*cusparseSpMatDescr_t descrA; 
	cusparseStatus_t stat;
	stat = cusparseCreateMatDescr(&descrA);
	CUSPARSE_CHECK(stat);
	stat = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	CUSPARSE_CHECK(stat);
	stat = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	CUSPARSE_CHECK(stat);*/

	/*Parameters*/
        int m = A.num_rows;
        int n = A.num_cols;
	int nnz = A.values.size();
	/* Creates handles for matrix */

	start= std::chrono::steady_clock::now();
        cusparseSpMatDescr_t descrA;
        cusparseStatus_t stat;
        stat = cusparseCreateCsr(&descrA,
                  n,
                  m,
                  nnz,
                  (void*) A_row_offsets,
                  (void*) A_column_indices,
                  (void*) A_values,
                  CUSPARSE_INDEX_32I,
                  CUSPARSE_INDEX_32I,
                  CUSPARSE_INDEX_BASE_ZERO,
                  CUDA_R_32F);
        CUSPARSE_CHECK(stat);

        end = std::chrono::steady_clock::now();
        auto parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Matrix init: " << parsingTime
                                << " μs" << std::endl;


	start = std::chrono::steady_clock::now();
	/* Creates handles for vectors*/
	cusparseDnVecDescr_t vecX, vecY;
	stat = cusparseCreateDnVec(&vecX, n, (void*) v, CUDA_R_32F);
        CUSPARSE_CHECK(stat);
       // cudaMalloc((void**) &u, sizeof(float) * (m));
        stat = cusparseCreateDnVec(&vecY, m, (void*) u, CUDA_R_32F);
        CUSPARSE_CHECK(stat);

        end = std::chrono::steady_clock::now();
        parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Vector init: " << parsingTime
                                << " μs" << std::endl;
	

	
	//start = std::chrono::steady_clock::now();
	//cusparseHandle_t handle;
	//stat = cusparseCreate(&handle);
        //end = std::chrono::steady_clock::now();
	CUSPARSE_CHECK(stat);	
        //end=std::chrono::steady_clock::now();
	size_t bufferSize;
	void *buffer = nullptr;
	float alpha = 1.0;
	float beta = 0.0;
	//start = std::chrono::steady_clock::now();
	CudaSparseCheck(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
        
	//end = std::chrono::steady_clock::now();
        //parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //std::cout << "Cusparse Handle: " << parsingTime
        //                        << " μs" << std::endl;


	/* Buffer creation and calculation */
	start= std::chrono::steady_clock::now();
	cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, &bufferSize);
        //end = std::chrono::steady_clock::now();

        //parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //std::cout << "Buffer size: " << parsingTime
        //                        << " μs" << std::endl;

	/*cusparseScsrgemm2_bufferSizeExt(handle, m, n, k, &alpha, descrA, A.num_entries, 
			A_row_offsets, A_column_indices, descrB, B.num_entries, B_row_offsets, B_column_indices, nullptr,
			descr_D, 0, nullptr, nullptr, info, &bufferSize);*/
	//start = std::chrono::steady_clock::now();
	cudaMalloc(&buffer, bufferSize);
        end = std::chrono::steady_clock::now();

        parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //std::cout << "Malloc: " << parsingTime
        //                        << " μs" << std::endl;
	std::cout << "Preprocesamiento cusp: " << parsingTime << " μs" << std::endl;
	file << std::to_string(parsingTime) + ","; //Preprocesamiento CUSP

        // step 3: compute u

	start = std::chrono::steady_clock::now();
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, buffer);
	cudaDeviceSynchronize();
	end = std::chrono::steady_clock::now();

       	parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Cusparse multiplication: " << parsingTime
                                << " μs" << std::endl;
	file << std::to_string(parsingTime) + ","; 

	start = std::chrono::steady_clock::now();
	cudaFree(buffer);
	end = std::chrono::steady_clock::now();
	parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "Free cusparse: " << parsingTime << " μs" << std::endl; 
	file << std::to_string(parsingTime) + ","; //Free CUSP

	// step 5: destroy the opaque structure
	//cusparseDestroyCsrgemm2Info(info);
}
