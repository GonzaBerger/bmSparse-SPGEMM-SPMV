#ifndef CUSPARSE_SPMV_CUH_
#define CUSPARSE_SPMV_CUH_


void cusparse_spmv(cusp::csr_matrix<int, float,cusp::device_memory> &A,
                float* v, float* u, cusparseHandle_t handle);


#endif /* CUSPARSE_SPMV_CUH_ */

