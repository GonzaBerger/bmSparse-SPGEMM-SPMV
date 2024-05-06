#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <thrust/tuple.h>

#include <thrust/device_vector.h>
//#include <thrust/pair.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <iostream>
#include "reader.h"
#include "half.hpp"
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include <chrono>
#include <thread>
#include "bmSpMatrix.h"
#include "cuda_profiler_api.h"
#include <cusp/transpose.h>
#include "bb_segsort-master/bb_segsort.h"
#include "cuSparse_spmv.cuh"
/* Dumping bmSparse matrices to disk */
#include <fstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <fstream>
#include <cusp/io/binary.h>
#include <sys/time.h>
#include <string>
#include <cmath>
/* Using tensor cores */
#include <mma.h>


#define BLOCK_WIDTH 8
#define BLOCK_HEIGHT 8
#define BMSP_BLOCK_SIZE (BLOCK_WIDTH * BLOCK_HEIGHT)
#define SHMEM_BLOCK_SIZE (BLOCK_WIDTH * BLOCK_HEIGHT)
#define BLOCK_SIZE (BLOCK_WIDTH * BLOCK_HEIGHT)
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 2//(BLOCK_SIZE / WARP_SIZE)
#define CUDA_BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)
#define INPUT_TYPE double
#define OUTPUT_TYPE float
#define FREE_VEC 1

/* Prints execution times and other relevant information */

// typedef these iterators for shorthand
typedef thrust::device_vector<uint64_t>::iterator uint64_it;

using namespace std;

struct is_same_row: public thrust::binary_function<uint64_t, uint64_t, bool> {
	__host__ __device__
	bool operator()(uint64_t key_1, uint64_t key_2) {
		return (key_1 >> 32) == (key_2 >> 32);
	}
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <class valueType>
inline __device__
void shmem_load(uint64_t bmp, valueType *shmem_ptr, valueType *values, const int lane_id, const int &address) {
	uint64_t my_bit = bmp & (uint64_t(1) << BMSP_BLOCK_SIZE - 1 - address);
	if (my_bit) {
		int pos = __popcll(bmp >> (BMSP_BLOCK_SIZE - address));
		shmem_ptr[lane_id] = values[pos];
	} else {
		shmem_ptr[lane_id] = 0;
	}
}

template <class ValueIn, class ValueOut>
__global__
void spmv_kernel_new(uint64_t* first_blocks, uint64_t* A_keys, ValueIn* A_values, ValueIn* v, ValueOut* u, uint64_t* A_bmps, uint64_t* A_offsets, int n){
	
	ValueOut res = 0;
	
	if(blockIdx.x*8+threadIdx.y >= n) return;

	uint64_t first = first_blocks[blockIdx.x], last = first_blocks[blockIdx.x+1];
	
	int blocksLeft = last - first;

	int blk = threadIdx.x/8;
	int col = threadIdx.x%8;
	int row = threadIdx.y;

	int lane_id = row*8 + col;

	while(blocksLeft>0){

		ValueIn aux, valv;

		uint64_t bmp, key, off;
		int blk_idx;

		aux=0;
		valv=0;

		if(blk<blocksLeft){

			blk_idx = last - blocksLeft + blk;
			bmp = A_bmps[blk_idx];
		
			uint64_t my_bit = bmp & (uint64_t(1) << BMSP_BLOCK_SIZE - 1 - lane_id);

			if(my_bit){
				off = A_offsets[blk_idx];
				key = (A_keys[blk_idx] << 32) >> 32;

				valv=v[key*8 + col];

				int pos = __popcll(bmp >> (BMSP_BLOCK_SIZE - lane_id));
				aux = A_values[off+pos];
			}else{
				aux = 0;
				valv = 0;
			}

		}

		res += (ValueOut) aux * valv;
		blocksLeft-=4;		
	}

	 __syncthreads();		

	for(int i=16; i>=1; i/=2){
		res += __shfl_down_sync(__activemask(), res, i, 32);
	}
	 __syncthreads();
	
	if( threadIdx.x == 0 ){
		u[blockIdx.x*8 + row] = res;
	}	
	

}


template <class ValueIn, class ValueOut>
__global__
void spmv_kernel(uint64_t* first_blocks, uint64_t* A_keys, ValueIn* A_values, ValueIn* v, ValueOut* u, uint64_t* A_bmps, uint64_t* A_offsets, int n){

	const int warp_id = threadIdx.x / 32;
		
	__shared__ ValueIn shmem [WARPS_PER_BLOCK*SHMEM_BLOCK_SIZE*2];
	ValueIn * blockA = &shmem[warp_id*SHMEM_BLOCK_SIZE];
	
	//ValueIn aux;
	ValueOut res = 0;
	const int lane_id = threadIdx.x % 64;
	uint64_t bmp;

	if(blockIdx.x*8+threadIdx.x/8 >= n)
		return;
	
	uint64_t first = first_blocks[blockIdx.x], last = first_blocks[blockIdx.x+1];

	for(uint64_t i = first; i < last; i++ ){
        	ValueIn *A_block_values = A_values + A_offsets[i];
		bmp = A_bmps[i];
		shmem_load<ValueIn>(bmp, blockA, A_block_values, lane_id, lane_id);
		res+= (ValueOut) (blockA[threadIdx.x]*v[((A_keys[i] << 32) >> 32)*8+threadIdx.x%8]);
	}

	// Reduccion de a 8
	for (int i=4; i>=1; i/=2)
        	res += __shfl_down_sync(__activemask(),res, i,8);

	__syncthreads();
	

	if((lane_id % 8) == 0)
		u[blockIdx.x*8+ threadIdx.x/8]	= res;

}
	
template <class ValueIn, class ValueOut>
inline void bmSparse_SpMV(bmSpMatrix<ValueIn> &A, ValueIn* v, ValueOut* u, bool batched){

	auto start = std::chrono::steady_clock::now();

	thrust::device_vector<uint64_t> A_offsets(A.keys.size()); 
	thrust::constant_iterator<uint64_t> ones_it(1);

	auto new_end = thrust::reduce_by_key(A.keys.begin(),
			A.keys.end(),
				ones_it,
				thrust::make_discard_iterator(),
				A_offsets.begin(),
				is_same_row());

	thrust::exclusive_scan(A_offsets.begin(), new_end.second+1, A_offsets.begin()); 

    uint64_t* offsets = thrust::raw_pointer_cast(A_offsets.data());

	uint64_t *A_keys_raw = thrust::raw_pointer_cast(A.keys.data());
	uint64_t *A_bmps_raw = thrust::raw_pointer_cast(A.bmps.data());
	ValueIn  *A_values_raw = thrust::raw_pointer_cast(A.values.data());
	uint64_t *first_values_A = thrust::raw_pointer_cast(A.offsets.data());
  	

	if(batched){
		spmv_kernel_new<<<A.num_cols/8 + 1, 64>>>(offsets, A_keys_raw, A_values_raw, v, u, A_bmps_raw, first_values_A, A.num_cols);
		//spmv_kernel_new<<<A.num_cols/8 + 1, dim3(32, 8)>>>(offsets, A_keys_raw, A_values_raw, v, u, A_bmps_raw, first_values_A, A.num_cols);
	}else{
		spmv_kernel<<<A.num_cols/8 + 1, 64>>>(offsets, A_keys_raw,/*columns,*/ A_values_raw, v, u, A_bmps_raw, first_values_A, A.num_cols);
	}

	cudaDeviceSynchronize();

	if(FREE_VEC){
		A_offsets.clear();
		A_offsets.shrink_to_fit();
	}

}

int main(int argc, char** argv) {
	/* Dump CSR and bmSpMatrix if needed */

	bool batched = false;

	cudaFree(0);	

	if(argc < 2){
		std::cout << "./main MatrixFolder A_Matrix" << std::endl;
		return 1;
	}

	if(argc > 2)
		batched = *argv[3] == '1';

	std::string A_path = std::string(argv[1]) + "/" + std::string(argv[2]);

	std::cout << "A matrix: " << A_path << std::endl;
	std::chrono::steady_clock::time_point start, end;


	/* Se leen matrices */
	cudaDeviceSynchronize();

	start = std::chrono::steady_clock::now();
	bmSpMatrix<half> A_bmSp(A_path + ".mtx", false);

	end = std::chrono::steady_clock::now();
	auto parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(
			end - start).count();
	std::cout << "Parsing mtx files / Loading matrices from disk BMSP: " << parsingTime
			<< " μs" << std::endl;
	cudaDeviceSynchronize();

	std::cout << "Running SpMV \n" ;

	OUTPUT_TYPE *v, *u, *vcpu;
		
	bmSpMatrix<OUTPUT_TYPE> A_matrix(A_path/* + ".mtx"*/, false);

	vcpu =(OUTPUT_TYPE *) malloc(sizeof(OUTPUT_TYPE)* A_matrix.num_cols);

	cudaFree(0);

	cudaMalloc((void**)&v, sizeof(OUTPUT_TYPE)* A_matrix.num_cols);
	cudaMalloc((void**)&u, sizeof(OUTPUT_TYPE)* A_matrix.num_rows);

	for(int i=0; i<A_matrix.num_cols;i++){
		vcpu[i] = 1;
	}

	start = std::chrono::steady_clock::now();

	cudaMemcpy(v, vcpu, sizeof(OUTPUT_TYPE)* A_matrix.num_cols,  cudaMemcpyHostToDevice);
	free(vcpu);
			
	end = std::chrono::steady_clock::now();
           	
	parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(
					end - start).count();
	std::cout << "Parsing mtx files / Loading matrix and vectors: " << parsingTime
					<< " μs" << std::endl;
	
	/*BmSparse*/
	cudaDeviceSynchronize();
	start = std::chrono::steady_clock::now();           

	bmSparse_SpMV(A_matrix, v, u, batched);
	cudaDeviceSynchronize();

	end = std::chrono::steady_clock::now();
	auto bmSpTime = std::chrono::duration_cast<std::chrono::microseconds>(
						end - start).count();
		
	std::cout << "bmSparse SpMV execution: " << bmSpTime << " μs" << std::endl;

	OUTPUT_TYPE *u_bmsp =(OUTPUT_TYPE *)  malloc(sizeof(OUTPUT_TYPE)* A_matrix.num_rows);
	cudaMemcpy(u_bmsp, u, sizeof(OUTPUT_TYPE)* A_matrix.num_rows,  cudaMemcpyDeviceToHost);

	return 0;
}
