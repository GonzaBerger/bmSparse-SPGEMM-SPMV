#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <thrust/tuple.h>

#include <thrust/device_vector.h>
#include <thrust/pair.h>
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
using namespace nvcuda;

#define BLOCK_WIDTH 8
#define BLOCK_HEIGHT 8
#define BMSP_BLOCK_SIZE (BLOCK_WIDTH * BLOCK_HEIGHT)
#define SHMEM_BLOCK_SIZE (BLOCK_WIDTH * BLOCK_HEIGHT)
#define TC_BLOCK_DIM 16
#define TC_BLOCK_SIZE (TC_BLOCK_DIM * TC_BLOCK_DIM)
#define BLOCK_SIZE (BLOCK_WIDTH * BLOCK_HEIGHT)
#define SHMEM_BLOCK_DIM 16
#define TASKS_PER_WARP 16
#define WARP_SIZE 32
#define TASK_BUFFER 8
#define WARPS_PER_BLOCK 2//(BLOCK_SIZE / WARP_SIZE)
#define CUDA_BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)
#define PARALLEL_BLOCKS 2
#define PARALLEL_TASKS 2
#define INPUT_TYPE float
#define OUTPUT_TYPE float
#define FREE_VEC 1
#define BORDER 2730000

#define MULTIPROCESSOR_COUNT 68 //2080 TI
#define BLOCK_COUNT (MULTIPROCESSOR_COUNT * 4)
/* Prints execution times and other relevant information */



struct task_list_elem {
	uint64_t first;
	uint64_t second;
};

// typedef these iterators for shorthand
typedef thrust::device_vector<uint64_t>::iterator uint64_it;

using namespace std;

struct is_same_row: public thrust::binary_function<uint64_t, uint64_t, bool> {
	__host__ __device__
	bool operator()(uint64_t key_1, uint64_t key_2) {
		return (key_1 >> 32) == (key_2 >> 32);
	}
};

struct is_same_ik: public thrust::binary_function<task_list_elem, task_list_elem, bool> {
	__device__
	bool operator()(task_list_elem fst, task_list_elem snd) {
		uint64_t fst_ik = (A_keys[fst.first] & 0xFFFFFFFF00000000) | (B_keys[fst.second] & 0x00000000FFFFFFFF);
		uint64_t snd_ik = (A_keys[snd.first] & 0xFFFFFFFF00000000) | (B_keys[snd.second] & 0x00000000FFFFFFFF);
		return fst_ik == snd_ik;
	}
	;
	uint64_t *A_keys, *B_keys;
};

struct is_less_ik: public thrust::binary_function<task_list_elem, task_list_elem, bool> {
	__device__
	bool operator()(task_list_elem fst, task_list_elem snd) {
		uint64_t fst_ik = (A_keys[fst.first] & 0xFFFFFFFF00000000) | (B_keys[fst.second] & 0x00000000FFFFFFFF);
		uint64_t snd_ik = (A_keys[snd.first] & 0xFFFFFFFF00000000) | (B_keys[snd.second] & 0x00000000FFFFFFFF);
		return fst_ik < snd_ik;
	}
	;
	uint64_t *A_keys, *B_keys;
};

struct idx_to_k: public thrust::binary_function<uint64_t, uint64_t, uint32_t> {
	__host__ __device__
	uint32_t operator()(thrust::tuple<uint64_t, uint64_t> key_position_pair, uint64_t idx) {
		uint64_t A_key = thrust::get<0>(key_position_pair);
		uint64_t col = (A_key << 32) >> 32;
		uint64_t position = thrust::get<1>(key_position_pair);
		task_list[position].second = pos[col] + idx;
		return (B_keys[pos[col] + idx] << 32) >> 32;
	};
	task_list_elem *task_list;
	uint64_t *pos;
	uint64_t *B_keys;
};


struct task_elem_to_C_key: public thrust::unary_function<task_list_elem, uint64_t> {
	__host__ __device__
	uint64_t operator()(task_list_elem elem) {
		return (A_keys[elem.first] & 0xFFFFFFFF00000000) | (B_keys[elem.second] & 0x00000000FFFFFFFF);
	}
	;
	uint64_t *A_keys;
	uint64_t *B_keys;
};

struct key_to_col: public thrust::unary_function<uint64_t, uint64_t> {
	__host__ __device__
	uint64_t operator()(uint64_t elem) {
		return (elem << 32) >> 32;
	}
	;
};

struct task_creator: public thrust::binary_function<uint64_t, uint32_t, task_list_elem> {
	__host__ __device__
	task_list_elem operator()(uint64_t A_pos, uint32_t idx) {
		task_list_elem task;
		task.first = A_pos;
		task.second = pos[A_keys[A_pos] & 0x00000000FFFFFFFF] + uint64_t(idx);
		return task;
	}

	uint64_t *A_keys;
	uint64_t *pos;
};

template <class valueType>
inline __device__
void shmem_load_new(uint64_t bmp, valueType *shmem_ptr, valueType *values, const int lane_id, const int &address) {
	uint64_t my_bit = bmp & (uint64_t(1) << BMSP_BLOCK_SIZE - 1 - address);
	if(my_bit) {
		int pos = __popcll(bmp >> (BMSP_BLOCK_SIZE - address));
		shmem_ptr[lane_id] = values[pos];
	} else {
		shmem_ptr[lane_id] = 0;
	}
	
}

template <class valueType>
inline __device__
void shmem_load(uint64_t bmp, valueType *shmem_ptr, valueType *values, const int lane_id, const int &address) {
	uint64_t my_bit = bmp & (uint64_t(1) << BMSP_BLOCK_SIZE - 1 - address);
	if (my_bit) {
		//int pos = bmp_elems - __popcll(bmp << threadIdx.x);
		int pos = __popcll(bmp >> (BMSP_BLOCK_SIZE - address));
//		if (blockIdx.x == 0 && threadIdx.x == 0) {
//			printf("Lane: %u, address: %u, bmp: %llu, pos: %u\n", lane_id, address, bmp, pos);
//		}
		shmem_ptr[lane_id] = values[pos];//ld_gbl_cg(values + pos);
	} else {
		shmem_ptr[lane_id] = 0;
	}
}


int cntBits(uint64_t data){
	int res=0;
	while(data){
		if(data&1){
			++res;
		}
		data>>=1;
	}
	return res;
}

template <class ValueIn, class ValueOut>
__global__
void spmv_kernel_new(uint64_t* first_blocks, uint64_t* A_keys, ValueIn* A_values, ValueIn* v, ValueOut* u, uint64_t* A_bmps, uint64_t* A_offsets, int n){

	// const int warp_id = threadIdx.x / 32;
	// int nnzRow = A_offsets[A_keys[last]] - A_offsets[A_keys[first]];
	// int nnzRow = A_offsets[last] - A_offsets[first];	
	// ValueIn *blockA;
	// ValueIn A_block_values2[8*32];	
	
	ValueOut res = 0;

	// __shared__ ValueOut shmem[9*32 + 1];
	
	if(blockIdx.x*8+threadIdx.y >= n) return;

	uint64_t first = first_blocks[blockIdx.x], last = first_blocks[blockIdx.x+1];
	
	int blocksLeft = last - first;

	int blk = threadIdx.x/8;
	int col = threadIdx.x%8;
	int row = threadIdx.y;

	int lane_id = row*8 + col;
	//int thread_id = threadIdx.y*blockDim.x+threadIdx.x;

	while(blocksLeft>0){

		ValueIn aux, valv;
		//if(isDense[last-blocksLeft+blk]){
				
			//int prevElems = A_offsets[last - blocksLeft];

			uint64_t bmp, key, off;
			int blk_idx;

			aux=0;
			valv=0;

			if(blk<blocksLeft){

				blk_idx = last - blocksLeft + blk;
				bmp = A_bmps[blk_idx];
		
				uint64_t my_bit = bmp & (uint64_t(1) << BMSP_BLOCK_SIZE - 1 - lane_id);

				if(my_bit){

				//blk_idx = last - blocksLeft + blk;
				//bmp = A_bmps[blk_idx];
					off = A_offsets[blk_idx];
					key = (A_keys[blk_idx] << 32) >> 32;

				// if(blockIdx.x*8 + row==57711 && col==0) printf("fila 57711, bmp=%.16llx blk=%d threadIdx.x=%d\n",A_bmps[blk_idx],blk,threadIdx.x);
				// if (threadIdx.y==0)
				// 	if(key*8 + col < n){
				// 		// shmem[8*32 + threadIdx.x] = v[((A_keys[last - blocksLeft] << 32) >> 32)*8 + threadIdx.x];
				// 		shmem[8*32 + threadIdx.x] = v[key*8 + col];
				// 	}else{
				// 		shmem[8*32 + threadIdx.x] = 0;
				// 	}

					valv=v[key*8 + col];

				//uint64_t my_bit = bmp & (uint64_t(1) << BMSP_BLOCK_SIZE - 1 - lane_id); 	
			
				//if(my_bit){
					int pos = __popcll(bmp >> (BMSP_BLOCK_SIZE - lane_id));
					//aux = A_values[A_offsets[blk_idx]+pos];
					aux = A_values[off+pos];
					// aux = shmem[off + pos - prevElems];
				}else{
					aux = 0;
					valv = 0;
				}

			}

		//}

		// __syncthreads();

		res += (ValueOut) aux * valv; // shmem[8*32 + threadIdx.x];
		// res += (ValueOut) shmem[threadIdx.y*32 + threadIdx.x]*shmem[8*32 + threadIdx.x];			//res += (ValueOut) shmem[threadIdx.x]*shmem[8*32 + blk*8 + col];
		blocksLeft-=4;
		//blocksLeft-=1;					
	}

	 __syncthreads();		

	for(int i=16; i>=1; i/=2){
		//acá también hay un cuello de botella importante
		res += __shfl_down_sync(__activemask(), res, i, 32);
	}
	 __syncthreads();
	
	if( threadIdx.x == 0 ){
		u[blockIdx.x*8 + row] = res;
	}	
	

}



//Version con dense
template <class ValueIn, class ValueOut>
__global__
void spmv_kernel_new_dense(uint64_t* first_blocks, uint64_t* A_keys, ValueIn* A_values, ValueIn* v, ValueOut* u, uint64_t* A_bmps, uint64_t* A_offsets, int n, bool* isDense, ValueIn* denseBlocks){

	ValueOut res = 0;
	if(blockIdx.x*8 + threadIdx.y >= n) return;
	
	uint64_t first = first_blocks[blockIdx.x], last = first_blocks[blockIdx.x + 1];

	int blocksLeft = last - first;
	int blk = threadIdx.x/8;
	int col = threadIdx.x%8;
	int row = threadIdx.y;

	int lane_id = row*8 + col;
	int thread_id = threadIdx.y*blockDim.x + threadIdx.x;
	int denseAmount = 0;	


//if(isDense[last-blocksLeft]){

	while(blocksLeft > 0){
		
		if(!isDense[last-blocksLeft+blk]){ 
		//if(41>40){	
	
			int prevElems = A_offsets[last - blocksLeft];
			uint64_t bmp, key, off;
			int blk_idx;
			ValueIn aux = 0;
			ValueIn valv = 0;

			if(blk<blocksLeft){
				blk_idx = last - blocksLeft + blk;
				bmp = A_bmps[blk_idx];
				off = A_offsets[blk_idx];
				key = (A_keys[blk_idx]<<32)>>32;
	
				valv = v[key*8 + col];
				uint64_t my_bit = bmp&(uint64_t(1)<<(BMSP_BLOCK_SIZE - 1 - lane_id));
		
				if(my_bit){
					int pos = __popcll(bmp>>(BMSP_BLOCK_SIZE - lane_id));
					aux = A_values[A_offsets[blk_idx] + pos];
				}else{
					aux = 0;
				}
				res += (ValueOut)aux*valv;
				//blocksLeft -= 4;

			}
		
		}else{
			if(blk<blocksLeft){
				uint64_t key = (A_keys[last-blocksLeft+blk]<<32)>>32;
				res += (ValueOut)denseBlocks[denseAmount*64 + row*8 + col]*v[key*8+col];		
				denseAmount++;
			}
		}

		blocksLeft -= 4;

	}

	for(int i=16; i>=1; i/=2){
		res += __shfl_down_sync(__activemask(), res, i, 32);
	}

	if(threadIdx.x == 0){
		u[blockIdx.x*8 + row] = res;
	}

//}

}



template <class ValueIn, class ValueOut>
__global__
void spmv_kernel_new_no_shmem(uint64_t* first_blocks, uint64_t* A_keys, ValueIn* A_values, ValueIn* v, ValueOut* u, uint64_t* A_bmps, uint64_t* A_offsets, int n){

	//const int warp_id = threadIdx.x / 32;
	
	ValueOut res = 0;
	//uint64_t bmp;

	//__shared__ ValueIn shmem[8*32 + 1];
	//__shared__ int shmemOffsets[4];
	//__shared__ uint64_t shmemBmps[4];
	//__shared__ uint64_t shmemCol; 
	
	if(blockIdx.x*8+threadIdx.x%8 >= n) return;

	uint64_t first = first_blocks[blockIdx.x], last = first_blocks[blockIdx.x+1];
	uint64_t blk, col, row, threadBitmap, threadCol;
	int threadOffset, lane_id;

	//int nnzRow = A_offsets[last] - A_offsets[first];	
	int blocksLeft = last - first;
	
	while(blocksLeft>0){
		blk = threadIdx.x/8;
		col = threadIdx.x%8;
		row = threadIdx.y;		
	
		lane_id = row*8 + col;

		threadOffset = A_offsets[last - blocksLeft + blk];
		threadBitmap = A_bmps[last - blocksLeft + blk];
		threadCol = (A_keys[last - blocksLeft] << 32) >> 32;
			
		int index;
		ValueIn aux;
		if(blk<blocksLeft){
			uint64_t my_bit = threadBitmap & (uint64_t(1) << BMSP_BLOCK_SIZE - 1 - lane_id%64); 	
			//uint64_t my_bit = A_bmps[last - blocksLeft + blk] & (uint64_t(1) << BMSP_BLOCK_SIZE - 1 - lane_id);	
			if(my_bit){
				int pos = __popcll(threadBitmap >> (BMSP_BLOCK_SIZE - lane_id%64));
				aux = A_values[threadOffset + pos];
			}else{
				aux = 0;
			}
		}else{
			aux = 0;
		}

		ValueIn dataVec;
		if(blk < blocksLeft){
			dataVec = v[threadCol*8 + blk*8 + col];
		}else{
			dataVec = 0;
		}
		
		res += (ValueOut) aux*dataVec;
		
		blocksLeft-=4;
					
	}

	//__syncthreads();		

	for(int i=16; i>=1; i/=2){
		res += __shfl_down_sync(__activemask(), res, i, 32);
	}
	__syncthreads();
	
	if( (lane_id % 8) == 0 ){
		u[blockIdx.x*8 + row] = res;
	}	
	//__syncthreads();

}

template <class ValueIn, class ValueOut>
__global__
void spmv_kernel_new_batched(uint64_t* first_blocks, uint64_t* A_keys, ValueIn* A_values, ValueIn* v, ValueOut* u, uint64_t* A_bmps, uint64_t* A_offsets, int n){

	const int warp_id = threadIdx.x / 32;
	
	ValueOut res = 0;
	uint64_t bmp;

	//__shared__ ValueIn shmem[9*32 + 1];
	__shared__ int shmemOffsets[32];
	__shared__ uint64_t shmemBmps[32];
	__shared__ uint64_t shmemVec[32];	

	int cont = 0;

	uint64_t first = first_blocks[blockIdx.x], last = first_blocks[blockIdx.x+1];
	uint64_t blk, col, row;
	//int nnzRow = A_offsets[A_keys[last]] - A_offsets[A_keys[first]];
	//int nnzRow = A_offsets[last] - A_offsets[first];	
	int blocksLeft = last - first;
	int lane_id;
	//ValueIn *blockA;
	//ValueIn A_block_values2[8*32];	

	while(blocksLeft>0){

		blk = threadIdx.x/8;
		col = threadIdx.x%8;
		row = threadIdx.y;

		lane_id = row*8 + col;

		if(cont%32 == 0){
			int minAux = (blocksLeft > 32) ? 32 : blocksLeft;
			if(threadIdx.x < minAux){
				shmemOffsets[threadIdx.x] = A_offsets[last - blocksLeft + threadIdx.x];
			}else if(threadIdx.x >= minAux && threadIdx.x < 2*minAux){
				shmemBmps[threadIdx.x - minAux] = A_bmps[last - blocksLeft + threadIdx.x - minAux];
			}else if(threadIdx.x >= 2*minAux && threadIdx.x < 3*minAux){
				shmemVec[threadIdx.x - 2*minAux] = v[((A_keys[last - blocksLeft + threadIdx.x - 2*minAux] << 32) >> 32)*8 + blk*8 + col];
			}
		}

		__syncthreads();

		ValueIn dataMat;
		if(blk<blocksLeft){
			bmp = shmemBmps[cont+blk];
			uint64_t my_bit = bmp & (uint64_t(1) << BMSP_BLOCK_SIZE - 1 - lane_id); 	
			
			if(my_bit){
				int pos = __popcll(bmp >> (BMSP_BLOCK_SIZE - lane_id));
				int index = shmemOffsets[cont+blk];
				dataMat = A_values[index+pos];
			}else{
				dataMat = 0;
			}
		}else{
			dataMat = 0;
		}

		ValueIn dataVec;		
		if(blk<blocksLeft){
			dataVec = shmemVec[cont+blk];
		}else{
			dataVec = 0;
		}

		res += (ValueOut) dataMat*dataVec;

		blocksLeft-=4;
		cont+=4;
		if(cont==32) cont=0;			
	}

	//__syncthreads();		

	for(int i=16; i>=1; i/=2){
		res += __shfl_down_sync(__activemask(), res, i, 32);
	}
	__syncthreads();
	
	if( (lane_id % 8) == 0 ){
		u[blockIdx.x*8 + row] = res;
	}	
	//__syncthreads();
	
}


template <class ValueIn, class ValueOut>
__global__
void spmv_kernel(uint64_t* first_blocks, uint64_t* A_keys,/*uint64_t* column_indices,*/ ValueIn* A_values, ValueIn* v, ValueOut* u, uint64_t* A_bmps, uint64_t* A_offsets, int n){

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
		//res+= (ValueOut) (aux)*(v[((A_keys[i] << 32) >> 32)*8 + threadIdx.x%8]);
	}

	// Reduccion de a 8
	for (int i=4; i>=1; i/=2)
        	res += __shfl_down_sync(__activemask(),res, i,8); // Falta optimizar la mascara

	__syncthreads();
	

	if((lane_id % 8) == 0)
		u[blockIdx.x*8+ threadIdx.x/8]	= res;

}

<<<<<<< HEAD

template <class ValueIn, class ValueOut>
__global__
void spmv_kernel_no_shmem(uint64_t* first_blocks, uint64_t* A_keys,/*uint64_t* column_indices,*/ ValueIn* A_values, ValueIn* v, ValueOut* u, uint64_t* A_bmps, uint64_t* A_offsets, int n){

        const int warp_id = threadIdx.x / 32;

        //__shared__ ValueIn shmem [WARPS_PER_BLOCK*SHMEM_BLOCK_SIZE*2];
        //ValueIn * blockA = &shmem[warp_id*SHMEM_BLOCK_SIZE];

	//ValueIn blocks [WARPS_PER_BLOCK*SHMEM_BLOCK_SIZE*2];
	//ValueIn * blockA = &blocks[warp_id*SHMEM_BLOCK_SIZE];

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

		// codigo del shmem load sin shmem
		uint64_t my_bit = bmp & (uint64_t(1) << BMSP_BLOCK_SIZE - 1 - lane_id);
	       
		ValueIn A_block_value;
		if (my_bit) {
	                int pos = __popcll(bmp >> (BMSP_BLOCK_SIZE - lane_id));
			//blockA[lane_id] = A_block_values[pos];
        		A_block_value = A_block_values[pos];
		} else {
                	//blockA[lane_id] = 0;
       	 		A_block_value = 0;
		}
		// fin codigo shmem

                //shmem_load<ValueIn>(bmp, blockA, A_block_values, lane_id, lane_id);
                //res+= (ValueOut) (blockA[threadIdx.x]*v[((A_keys[i] << 32) >> 32)*8+threadIdx.x%8]);
               	res+= (ValueOut) A_block_value*v[((A_keys[i] << 32) >> 32)*8+threadIdx.x%8];
		//res+= (ValueOut) (aux)*(v[((A_keys[i] << 32) >> 32)*8 + threadIdx.x%8]);
        }

        // Reduccion de a 8
        for (int i=4; i>=1; i/=2)
                res += __shfl_down_sync(__activemask(),res, i,8); // Falta optimizar la mascara

        __syncthreads();


        if((lane_id % 8) == 0)
                u[blockIdx.x*8+ threadIdx.x/8]  = res;
}





=======
>>>>>>> 7a2e3211c04a7e2059e8279160648d6e25bce12e
template <class ValueIn, class ValueOut>
__global__
void spmv_kernel_equitative(uint64_t* first_blocks, uint64_t* A_keys,/*uint64_t* column_indices,*/ ValueIn* A_values, ValueIn* v, ValueOut* u, uint64_t* A_bmps, uint64_t* A_offsets, int n, int blksPerCudaBlock){
	
	const int warp_id = threadIdx.x / 32;
		
	__shared__ ValueIn shmem [WARPS_PER_BLOCK*SHMEM_BLOCK_SIZE*2];
	ValueIn * blockA = &shmem[warp_id*SHMEM_BLOCK_SIZE];
	
	//ValueIn aux;
	ValueOut res, acum = 0;
	const int lane_id = threadIdx.x % 64;
	uint64_t bmp;

	if(blockIdx.x*8+threadIdx.x/8 >= n)
		return;
	
	//uint64_t first = first_blocks[blockIdx.x], last = first_blocks[blockIdx.x+1];

	uint64_t first = blksPerCudaBlock*blockIdx.x, last = first + blksPerCudaBlock;
	uint64_t previousRow = A_keys[blksPerCudaBlock*blockIdx.x] >> 32, currentRow;

	for(uint64_t i = first; i < last; i++ ){

        	ValueIn *A_block_values = A_values + A_offsets[i]; 
		bmp = A_bmps[i]; 
		shmem_load<ValueIn>(bmp, blockA, A_block_values, lane_id, lane_id);
		
		currentRow = A_keys[i] >> 32;
		if( i==last-1 || (previousRow != currentRow) ){
			// Reduccion de a 8
			for (int i=4; i>=1; i/=2)
		        	res += __shfl_down_sync(__activemask(), res, i, 8);

			__syncthreads();
			
			if((lane_id % 8) == 0){
				//atomicAdd(u[blockIdx.x*8+ threadIdx.x/8], res);
				u[blockIdx.x*8 + threadIdx.x/8] = res;
			}

			res = 0;
			previousRow = A_keys[i] >> 32;
			res+= (ValueOut) (blockA[threadIdx.x]*v[((A_keys[i] << 32) >> 32)*8+threadIdx.x%8]);
	
		}

	}
}

	
template <class ValueIn, class ValueOut>
inline void bmSparse_SpMV(bmSpMatrix<ValueIn> &A, ValueIn* v, ValueOut* u, std::ofstream& file){

	//int device;
	//cudaGetDevice(&device);

	//struct cudaDeviceProp props;
	//cudaGetDeviceProperties(&props, device);

	//std::cout << props.name << " multiprocessors: " << props.multiProcessorCount << std::endl;

	auto start = std::chrono::steady_clock::now();

        //Usar las keys para reducir y saber cuantos bloques por fila
        thrust::device_vector<uint64_t> A_offsets(A.keys.size()); 
        thrust::constant_iterator<uint64_t> ones_it(1);
	
	//auto end = std::chrono::steady_clock::now();
	//auto parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
	//std::cout << "Dev vector 1: " << parsingTime << " μs" << std::endl;

	//start = std::chrono::steady_clock::now();

        auto new_end = thrust::reduce_by_key(A.keys.begin(),
        		A.keys.end(),
                	ones_it,
                	thrust::make_discard_iterator(),
                	A_offsets.begin(),
                	is_same_row());

	//end = std::chrono::steady_clock::now();
	//parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
	//std::cout << "Reduce by key: " << parsingTime << " μs" << std::endl;	

	//start = std::chrono::steady_clock::now();

	thrust::exclusive_scan(A_offsets.begin(), new_end.second+1, A_offsets.begin()); 

	//end = std::chrono::steady_clock::now();
        //parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
        //std::cout << "Exclusive scan: " << parsingTime << " μs" << std::endl;

	//start = std::chrono::steady_clock::now();

	//thrust::device_vector<uint64_t> A_indices(A.keys.size());
        //thrust::device_vector<uint64_t> A_columns(A.keys.size());     //reuso indices
        //thrust::transform(A.keys.begin(), A.keys.end(), A_indices.begin(),
        //                key_to_col());

	auto end = std::chrono::steady_clock::now();
        auto parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
        std::cout << "Preprocesamiento: " << parsingTime << " μs" << std::endl;
	file << std::to_string(parsingTime) + ","; 	

	//start = std::chrono::steady_clock::now();

        //uint64_t* columns = thrust::raw_pointer_cast(A_indices.data());
        uint64_t* offsets = thrust::raw_pointer_cast(A_offsets.data());

	uint64_t *A_keys_raw = thrust::raw_pointer_cast(A.keys.data());
        uint64_t *A_bmps_raw = thrust::raw_pointer_cast(A.bmps.data());
        ValueIn  *A_values_raw = thrust::raw_pointer_cast(A.values.data());
        uint64_t *first_values_A = thrust::raw_pointer_cast(A.offsets.data());

	//end = std::chrono::steady_clock::now();
        //parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
        //std::cout << "Raw pointers: " << parsingTime << " μs" << std::endl;

	//auto end = std::chrono::steady_clock::now();
	//auto parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
	//std::cout << "Preprocesamiento: " << parsingTime << " μs" << std::endl;
	//file << std::to_string(parsingTime) + ","; Preprocesamiento BMSP



	//TESTEO DE GEMM

	//thrust::device_ptr<uint64_t> ptrBmps = A.bmps.data();

	//bool isDense[A.bmps.size()];
	/*bool isDense[A.]
	int denseAmount = 0;
	for(int j=0; j<A.bmps.size(); j++){
		if(cntBits(A.bmps[j]) > 40){
			isDense[j] = true;	
			denseAmount++;
		}else{
			isDense[j] = false;
		}
	}*/

/*
	float denseBlocks[denseAmount*64];
	

	int denseCount = 0;
	for(int k=0; k<A.bmps.size(); k++){
		if(isDense[k]){
			for(int l=1; l<=64; l++){
				if(A.bmps.data()[k]&(uint64_t(1)<<(64-l)) == 0){
					denseBlocks[denseCount*64 + l-1] = 0;
				}else{
					int offset = 0;
					if(k>0) offset=A.offsets.data()[k-1];
					denseBlocks[denseCount*64 + l-1] = A.values[offset + l-1];
				}

			}

			denseCount++;
		}
	}

	bool *isDenseRaw = thrust::raw_pointer_cast(isDense);
	ValueIn *denseBlocksRaw = thrust::raw_pointer_cast(denseBlocks);
*/
	//TESTEO DE GEMM


  	start = std::chrono::steady_clock::now();

	//const int spmvShmemSize = (WARPS_PER_BLOCK*SHMEM_BLOCK_SIZE*2)*sizeof(ValueIn);
	//, spmvShmemSize
  	

	spmv_kernel<<<A.num_cols/8 + 1, 64>>>(offsets, A_keys_raw,/*columns,*/ A_values_raw, v, u, A_bmps_raw, first_values_A, A.num_cols);
	//cudaDeviceSynchronize();



	//Version original
	//spmv_kernel_new<<<A.num_cols/8 + 1, 64>>>(offsets, A_keys_raw, A_values_raw, v, u, A_bmps_raw, first_values_A, A.num_cols);

	//Version con dense
	//spmv_kernel_new_dense<<<A.num_cols/8 + 1, 64>>>(offsets, A_keys_raw, A_values_raw, v, u, A_bmps_raw, first_values_A, A.num_cols, isDenseRaw, denseBlocksRaw);
		
	//spmv_kernel_new<<<A.num_cols/8 + 1, dim3(32, 8)>>>(offsets, A_keys_raw, A_values_raw, v, u, A_bmps_raw, first_values_A, A.num_cols);
	
	//int blksPerCudaBlock = A.keys.size()/BLOCK_COUNT;
	//spmv_kernel_equitative<<<BLOCK_COUNT + 1, 64>>>(offsets, A_keys_raw, A_values_raw, v, u, A_bmps_raw, first_values_A, A.num_cols, blksPerCudaBlock);
	cudaDeviceSynchronize();
	
	end = std::chrono::steady_clock::now();
        parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
        std::cout << "kernel: " << parsingTime << " μs" << std::endl;
	file << std::to_string(parsingTime) + ",";

	start = std::chrono::steady_clock::now();
	
	if(FREE_VEC){
		A_offsets.clear();
		A_offsets.shrink_to_fit();
		//A_indices.clear();
		//A_indices.shrink_to_fit();
	}

	end = std::chrono::steady_clock::now();
        parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
        std::cout << "Free vectors: " << parsingTime << " μs" << std::endl;
	file << std::to_string(parsingTime) + ","; //Free Vectors BMSP
	
	/*
	int acum = 0, aux = 0, prom = A.values.size()/A.num_rows;
	//for(int i=0; i<A.num_rows-1; i++){
	//	aux = (A.offsets[A_offsets[i+1]] - A.offsets[A_offsets[i]]) - prom;
	//	acum += aux*aux;
	//}
	//aux = (A-values.size() - A.offsets[A_offsets[A.num_rows-1]]) - prom;
	//acum += aux*aux;
	
	uint64_t row = A.keys[0] >> 32;
	for(int i=0; i<A.keys.size()-1; i++){
		if(A.keys[i] >> 32 == row){
			//aux += A_offsets[0];
			aux += A.offsets[i+1] - A.offsets[i];
		}else{
			row = A.keys[i] >> 32;
			aux -= prom;
			acum += aux*aux;
			aux = 0;
		}
	}
	aux += A.values.size() - A.offsets[A.keys.size() - 1];
	aux -= prom;
	acum += aux*aux;

	double stdDeviation = sqrt( (double) acum/A.num_rows );
	std::cout << "Standard deviation of matrix: " << stdDeviation << std::endl;
	file << std::to_string(stdDeviation);

	*/

}

/* C_keys es el ayout de C
 size es la cantidad de keys de C
 task_list es una lista que de pares de bitmaps, cada par constituye una multiplicación de bloques a hacer
 first_task indica en el i-esimo elemento, el índice del primer elemento en la task_list asociado al bloque i + 1 de C.
 En la práctica, esos índices se multiplican por 2 por cómo está implementada la task list.
 first_value_A indica en el i-esimo elemento, el indice del primer elemento del bloque i en values_A

 */


template <class valueType>
inline __device__
void load_inputs(wmma::fragment<wmma::matrix_a, 16, 16, 16, valueType, wmma::row_major> &a_frag,
		wmma::fragment<wmma::matrix_b, 16, 16, 16, valueType, wmma::row_major> &b_frag,
		const int frag_A_index, const int frag_B_index, const int frag_start, const int frag_start_t,
		const int task, uint64_t * const bmps, uint64_t * const first_values,
		valueType * const A_values, valueType * const B_values) {

	uint64_t bmp_A = bmps[task];
	uint64_t bmp_B = bmps[task + 1];

	valueType *A_block_values = A_values + first_values[task];
	valueType *B_block_values = B_values + first_values[task + 1];

	/* Loads matrix A */
	uint64_t my_bit = bmp_A & (1ULL << BLOCK_SIZE - 1 - frag_start);
	if (my_bit) {
		int pos = __popcll(bmp_A >> (BLOCK_SIZE - frag_start));
		a_frag.x[frag_A_index] = A_block_values[pos];//ld_gbl_cg(values + pos);
	}
	my_bit = bmp_A & (1ULL << BLOCK_SIZE - 1 - (frag_start + 1));
	if (my_bit) {
		int pos = __popcll(bmp_A >> (BLOCK_SIZE - (frag_start + 1)));
		a_frag.x[frag_A_index + 1] = A_block_values[pos];//ld_gbl_cg(values + pos);
	}

	/* Loads matrix B */
	my_bit = bmp_B & (1ULL << BLOCK_SIZE - 1 - frag_start_t);
	if (my_bit) {
		int pos = __popcll(bmp_B >> (BLOCK_SIZE - frag_start_t));
		b_frag.x[frag_B_index] = B_block_values[pos];//ld_gbl_cg(values + pos);
	}
	my_bit = bmp_B & (1ULL << BLOCK_SIZE - 1 - (frag_start_t + 1));
	if (my_bit) {
		int pos = __popcll(bmp_B >> (BLOCK_SIZE - (frag_start_t + 1)));
		b_frag.x[frag_B_index + 1] = B_block_values[pos];//ld_gbl_cg(values + pos);
	}
}


//Two blocks two tasks
template <class valueIn, class valueOut>
__global__
__launch_bounds__(64, 32)
void multiplyV15(uint64_t* task_list, uint64_t *first_task,
		valueIn *A_values, valueIn *B_values, valueOut *C_values, uint64_t *first_values_A, uint64_t *first_values_B,
              uint64_t *A_bmps, uint64_t *B_bmps, uint64_t *C_bmps, uint64_t *offsets, uint64_t size// uint64_t *A_keys, uint64_t *C_keys
              ) {

	extern __shared__  valueIn shmem[];
	uint64_t cur_task, last_task;


	const int lane_id = threadIdx.x % 32;
	const int warp_id = threadIdx.x / 32;
	const int t_lane_id = (lane_id % 8) * 8 + (lane_id / 8);

	valueIn *block_A = &shmem[warp_id * SHMEM_BLOCK_SIZE];
	valueIn *block_B = &block_A[WARPS_PER_BLOCK * SHMEM_BLOCK_SIZE];
	uint64_t *bmps = (uint64_t*)&shmem[2 * WARPS_PER_BLOCK * SHMEM_BLOCK_SIZE]
			+ warp_id * TASKS_PER_WARP * 2;
	uint64_t *first_values = &bmps[WARPS_PER_BLOCK * TASKS_PER_WARP * 2];

	for (int C_block = blockIdx.x * WARPS_PER_BLOCK + warp_id; C_block < size; C_block += gridDim.x * WARPS_PER_BLOCK) {

		if (C_block == 0) {
			cur_task = 0;
		} else {
			cur_task = first_task[C_block - 1];
		}

		last_task = first_task[C_block];

		valueOut result[2] = { };
		for (; cur_task < last_task; cur_task += TASKS_PER_WARP) {
			int pos = cur_task * 2 + lane_id;
			if (pos < last_task * 2) {
				uint64_t task = task_list[pos];
				if (lane_id % 2 == 0) {
					bmps[lane_id] = A_bmps[task];
					first_values[lane_id] = first_values_A[task];
				} else {
					bmps[lane_id] = B_bmps[task];
					first_values[lane_id] = first_values_B[task];
				}
			}

			__syncwarp();

			for (int i = 0; (i < TASKS_PER_WARP * 2) && (cur_task + i/2 < last_task); i += 2) {
				uint64_t bmp_A = bmps[i];
				uint64_t bmp_B = bmps[i + 1];

				valueIn *A_block_values = A_values + first_values[i];
				valueIn *B_block_values = B_values + first_values[i + 1];
				/* Cargo en memoria compartida los valores */
				__syncwarp();
				shmem_load<valueIn>(bmp_A, block_A, A_block_values, lane_id, lane_id);
				shmem_load<valueIn>(bmp_A, block_A + WARP_SIZE, A_block_values, lane_id, lane_id + WARP_SIZE);
				shmem_load<valueIn>(bmp_B, block_B, B_block_values, lane_id, t_lane_id);
				shmem_load<valueIn>(bmp_B, block_B + WARP_SIZE, B_block_values, lane_id, t_lane_id + BLOCK_WIDTH / 2);
				__syncwarp();

				/* Multiplico el bloque de A por el de B y acumulo en out */

				for (int t = 0; t < 8; t++) {
					valueIn B_value = block_B[t * 8 + (lane_id % 8)];
					result[0] += (valueOut)(block_A[(lane_id / 8) * 8 + t] * B_value);
					result[1] += (valueOut)(block_A[(lane_id / 8) * 8 + t + WARP_SIZE] * B_value);
				}

	//			float sum = 0;
	////			unsigned char row = *((unsigned char *)&bmp_A + 7 - threadIdx.x / 8);
	////			if (*((unsigned char *)&bmp_A + 7 - threadIdx.x / 8)) {
	//				for (int t = 0; t < 8; t++) {
	//	//				if (row & (0x80 >> t)) {
	//					sum += block_A[(threadIdx.x / 8) * 8 + t] * block_B[t * 8 + (threadIdx.x % 8)];
	//	//				}
	//				}
	//				out[threadIdx.x] += sum;
	////			}

			}
		}


//		/* DEBUG */
//		__syncthreads();
//		if (C_block == 100182 && threadIdx.x == 7) {
//			printf("\n");
//			for (int i = 0; i < 64; i++) {
//				if (i % 8 == 0) {
//					printf("\n");
//				}
//				printf("%f ", out[i]);
//			}
//			printf("\n");
//		}
//
		if (C_bmps[C_block] & (uint64_t(1) << BMSP_BLOCK_SIZE - 1 - lane_id)) {
			int pos = __popcll(C_bmps[C_block] >> (BMSP_BLOCK_SIZE - lane_id));
			C_values[offsets[C_block] + pos] =
					result[0];
		}
		if (C_bmps[C_block] & (uint64_t(1) << BMSP_BLOCK_SIZE - 1 - (lane_id + WARP_SIZE))) {
			int pos = __popcll(C_bmps[C_block] >> (BMSP_BLOCK_SIZE - (lane_id + WARP_SIZE)));
			C_values[offsets[C_block] + pos] =
					result[1];
		}

	}

}


template <class valueIn, class valueOut>
__global__
void multiplyV14(uint64_t* task_list, uint64_t *first_task,
		valueIn *A_values, valueIn *B_values, valueOut *C_values, uint64_t *first_values_A, uint64_t *first_values_B,
              uint64_t *A_bmps, uint64_t *B_bmps, uint64_t *C_bmps, uint64_t *offsets, uint64_t size// uint64_t *A_keys, uint64_t *C_keys
              ) {

	extern __shared__  valueIn shmem[];
	uint64_t cur_task[PARALLEL_BLOCKS], last_task[PARALLEL_BLOCKS];

	const int lane_id = threadIdx.x % 32;
	const int half_lane_id = lane_id % 16;
	const int warp_id = threadIdx.x / 32;
	const int half_warp_id = lane_id / 16;
	const int frag_start = lane_id * 2;
	/* This takes into account not only the different mapping for B fragment
	 * but the fact that the bitmap for B is transposed.
	 */
	const int frag_start_t = 8 * (lane_id / 4) + 2 * (lane_id % 4);

	uint64_t *bmps = (uint64_t*)&shmem + warp_id * PARALLEL_BLOCKS * TASK_BUFFER * 2;
	uint64_t *first_values = &bmps[WARPS_PER_BLOCK * PARALLEL_BLOCKS * TASK_BUFFER * 2];

	/* Initialize input matrices to zero */
//	for (int i = threadIdx.x; i < 2 * WARPS_PER_BLOCK * SHMEM_BLOCK_SIZE; i += blockDim.x) {
//		shmem[i] = 0;
//	}
//	__syncthreads();
	int C_block = blockIdx.x * WARPS_PER_BLOCK * PARALLEL_BLOCKS + warp_id * PARALLEL_BLOCKS;
	for (; C_block < size; C_block += gridDim.x * WARPS_PER_BLOCK * PARALLEL_BLOCKS) {

		if (C_block == 0) {
			cur_task[0] = 0;
		} else {
			cur_task[0] = first_task[C_block - 1];
		}
		last_task[0] = first_task[C_block];

		if (C_block + 1 < size) {
			cur_task[1] = first_task[C_block];
			last_task[1] = first_task[C_block + 1];
		} else {
			/* If size is odd, last_task <= cur_task, indicating that there are no tasks
			 * for the last even block (since it doesn't exist)
			 */
			cur_task[1] = 0;
			last_task[1] = 0;
		}

		wmma::fragment<wmma::matrix_a, 16, 16, 16, valueIn, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, valueIn, wmma::row_major> b_frag;
		wmma::fragment<wmma::accumulator, 16, 16, 16, valueOut> c_frag;
		wmma::fill_fragment(c_frag, 0.0f);

		for (; cur_task[0] < last_task[0] || cur_task[1] < last_task[1]; cur_task[0] += TASK_BUFFER, cur_task[1] += TASK_BUFFER) {
			int pos = cur_task[half_warp_id] * 2 + half_lane_id;
			if (pos < last_task[half_warp_id] * 2) {
				uint64_t task = task_list[pos];
				if (lane_id % 2 == 0) {
					bmps[lane_id] = A_bmps[task];
					first_values[lane_id] = first_values_A[task];
				} else {
					bmps[lane_id] = B_bmps[task];
					first_values[lane_id] = first_values_B[task];
				}
			}

			//(__any_sync(0xFFFFFFFF, cur_task < last_task)
//			bool tasks_remaining = true;
			for (int i = 0; (i < TASK_BUFFER) && (cur_task[0] + i < last_task[0] || cur_task[1] + i < last_task[1]); i += PARALLEL_TASKS) {
				wmma::fill_fragment(a_frag, 0.0f);
				wmma::fill_fragment(b_frag, 0.0f);

				/* First block, first task */
				if (cur_task[0] + i < last_task[0]) {
					load_inputs<valueIn>(a_frag, b_frag, 0, 0, frag_start, frag_start_t, 2 * i, bmps, first_values,
							A_values, B_values);
				}

				/* First block, second task */
				if (cur_task[0] + i + 1 < last_task[0]) {
					load_inputs<valueIn>(a_frag, b_frag, 4, 2, frag_start, frag_start_t, 2 * (i + 1), bmps, first_values,
							A_values, B_values);
				}

				/* Second block, first task */
				if (cur_task[1] + i < last_task[1]) {
					load_inputs<valueIn>(a_frag, b_frag, 2, 4, frag_start, frag_start_t, 2 * TASK_BUFFER + 2 * i,
							bmps, first_values, A_values, B_values);
				}

				/* Second block, second task */
				if (cur_task[1] + i + 1 < last_task[1]) {
					load_inputs<valueIn>(a_frag, b_frag, 6, 6, frag_start, frag_start_t, 2 * TASK_BUFFER + 2 * (i + 1),
							bmps, first_values, A_values, B_values);
				}

				wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

			}
		}

//		if (blockIdx.x == 0 && threadIdx.x == 0) {
//			printf("\n");
//			for (int i = 0; i < 16 * 16; i++) {
//				if (i % 16 == 0) {
//					printf("\n");
//				}
//				printf("%f ", (float)out[i]);
//			}
//		}
//		__syncwarp();

		if (C_bmps[C_block] & (uint64_t(1) << BLOCK_SIZE - 1 - frag_start)) {
			int pos = __popcll(C_bmps[C_block] >> (BLOCK_SIZE - frag_start));
			C_values[offsets[C_block] + pos] =
							c_frag.x[0];
		}
		if (C_bmps[C_block] & (uint64_t(1) << BLOCK_SIZE - 1 - (frag_start + 1))) {
			int pos = __popcll(C_bmps[C_block] >> (BLOCK_SIZE - (frag_start + 1)));
			C_values[offsets[C_block] + pos] =
							c_frag.x[1];
		}

		if (C_block + 1 < size) {
			if (C_bmps[C_block + 1] & (uint64_t(1) << BLOCK_SIZE - 1 - frag_start)) {
				int pos = __popcll(C_bmps[C_block + 1] >> (BLOCK_SIZE - frag_start));
				C_values[offsets[C_block + 1] + pos] =
								c_frag.x[6];
			}
			if (C_bmps[C_block + 1] & (uint64_t(1) << BLOCK_SIZE - 1 - (frag_start + 1))) {
				int pos = __popcll(C_bmps[C_block + 1] >> (BLOCK_SIZE - (frag_start + 1)));
				C_values[offsets[C_block + 1] + pos] =
								c_frag.x[7];
			}
		}

	}

}


//Two blocks
template <class valueIn, class valueOut>
__global__
void multiplyV13(uint64_t* task_list, uint64_t *first_task,
		valueIn *A_values, valueIn *B_values, valueOut *C_values, uint64_t *first_values_A, uint64_t *first_values_B,
              uint64_t *A_bmps, uint64_t *B_bmps, uint64_t *C_bmps, uint64_t *offsets, uint64_t size// uint64_t *A_keys, uint64_t *C_keys
              ) {

	extern __shared__  valueIn shmem[];
	uint64_t cur_task[PARALLEL_BLOCKS], last_task[PARALLEL_BLOCKS];

	const int lane_id = threadIdx.x % 32;
	const int half_lane_id = lane_id % 16;
	const int warp_id = threadIdx.x / 32;
	const int half_warp_id = lane_id / 16;
	const int frag_start = lane_id * 2;
	/* This takes into account not only the different mapping for B fragment
	 * but the fact that the bitmap for B is transposed.
	 */
	const int frag_start_t = 8 * (lane_id / 4) + 2 * (lane_id % 4);

	uint64_t *bmps = (uint64_t*)&shmem + warp_id * PARALLEL_BLOCKS * TASK_BUFFER * 2;
	uint64_t *first_values = &bmps[WARPS_PER_BLOCK * PARALLEL_BLOCKS * TASK_BUFFER * 2];

	/* Initialize input matrices to zero */
//	for (int i = threadIdx.x; i < 2 * WARPS_PER_BLOCK * SHMEM_BLOCK_SIZE; i += blockDim.x) {
//		shmem[i] = 0;
//	}
//	__syncthreads();
	int C_block = blockIdx.x * WARPS_PER_BLOCK * PARALLEL_BLOCKS + warp_id * PARALLEL_BLOCKS;
	for (; C_block < size; C_block += gridDim.x * WARPS_PER_BLOCK * PARALLEL_BLOCKS) {

		if (C_block == 0) {
			cur_task[0] = 0;
		} else {
			cur_task[0] = first_task[C_block - 1];
		}
		last_task[0] = first_task[C_block];

		if (C_block + 1 < size) {
			cur_task[1] = first_task[C_block];
			last_task[1] = first_task[C_block + 1];
		} else {
			/* If size is odd, last_task <= cur_task, indicating that there are no tasks
			 * for the last even block (since it doesn't exist)
			 */
			cur_task[1] = 0;
			last_task[1] = 0;
		}

		wmma::fragment<wmma::matrix_a, 16, 16, 16, valueIn, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, valueIn, wmma::row_major> b_frag;
		wmma::fragment<wmma::accumulator, 16, 16, 16, valueOut> c_frag;
		wmma::fill_fragment(c_frag, 0.0f);

		for (; cur_task[0] < last_task[0] || cur_task[1] < last_task[1]; cur_task[0] += TASK_BUFFER, cur_task[1] += TASK_BUFFER) {
			int pos = cur_task[half_warp_id] * 2 + half_lane_id;
			if (pos < last_task[half_warp_id] * 2) {
				uint64_t task = task_list[pos];
				if (lane_id % 2 == 0) {
					bmps[lane_id] = A_bmps[task];
					first_values[lane_id] = first_values_A[task];
				} else {
					bmps[lane_id] = B_bmps[task];
					first_values[lane_id] = first_values_B[task];
				}
			}

			//(__any_sync(0xFFFFFFFF, cur_task < last_task)
//			bool tasks_remaining = true;
			for (int i = 0; (i < TASK_BUFFER) && (cur_task[0] + i < last_task[0] || cur_task[1] + i < last_task[1]); i++) {
				wmma::fill_fragment(a_frag, 0.0f);
				wmma::fill_fragment(b_frag, 0.0f);

				/* First block */
				if (cur_task[0] + i < last_task[0]) {
					load_inputs<valueIn>(a_frag, b_frag, 0, 0, frag_start, frag_start_t, 2 * i, bmps, first_values,
							A_values, B_values);
				}

				/* Second block */
				if (cur_task[1] + i < last_task[1]) {
					load_inputs<valueIn>(a_frag, b_frag, 6, 6, frag_start, frag_start_t, 2 * TASK_BUFFER + 2 * i,
							bmps, first_values, A_values, B_values);
				}

				wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

			}
		}

//		if (blockIdx.x == 0 && threadIdx.x == 0) {
//			printf("\n");
//			for (int i = 0; i < 16 * 16; i++) {
//				if (i % 16 == 0) {
//					printf("\n");
//				}
//				printf("%f ", (float)out[i]);
//			}
//		}
//		__syncwarp();

		if (C_bmps[C_block] & (uint64_t(1) << BLOCK_SIZE - 1 - frag_start)) {
			int pos = __popcll(C_bmps[C_block] >> (BLOCK_SIZE - frag_start));
			C_values[offsets[C_block] + pos] =
							c_frag.x[0];
		}
		if (C_bmps[C_block] & (uint64_t(1) << BLOCK_SIZE - 1 - (frag_start + 1))) {
			int pos = __popcll(C_bmps[C_block] >> (BLOCK_SIZE - (frag_start + 1)));
			C_values[offsets[C_block] + pos] =
							c_frag.x[1];
		}

		if (C_block + 1 < size) {
			if (C_bmps[C_block + 1] & (uint64_t(1) << BLOCK_SIZE - 1 - frag_start)) {
				int pos = __popcll(C_bmps[C_block + 1] >> (BLOCK_SIZE - frag_start));
				C_values[offsets[C_block + 1] + pos] =
								c_frag.x[6];
			}
			if (C_bmps[C_block + 1] & (uint64_t(1) << BLOCK_SIZE - 1 - (frag_start + 1))) {
				int pos = __popcll(C_bmps[C_block + 1] >> (BLOCK_SIZE - (frag_start + 1)));
				C_values[offsets[C_block + 1] + pos] =
								c_frag.x[7];
			}
		}

	}

}


//Direct frag access
template <class valueIn, class valueOut>
__global__
void multiplyV12(uint64_t* task_list, uint64_t *first_task,
		valueIn *A_values, valueIn *B_values, valueOut *C_values, uint64_t *first_values_A, uint64_t *first_values_B,
              uint64_t *A_bmps, uint64_t *B_bmps, uint64_t *C_bmps, uint64_t *offsets, uint64_t size// uint64_t *A_keys, uint64_t *C_keys
              ) {

	extern __shared__  valueIn shmem[];
	uint64_t cur_task, last_task;

	const int lane_id = threadIdx.x % 32;
	const int warp_id = threadIdx.x / 32;
	const int frag_start = lane_id * 2;
	/* This takes into account not only the different mapping for B fragment
	 * but the fact that the bitmap for B is transposed.
	 */
	const int frag_start_t = 8 * (lane_id / 4) + 2 * (lane_id % 4);

	uint64_t *bmps = (uint64_t*)&shmem + warp_id * TASKS_PER_WARP * 2;
	uint64_t *first_values = &bmps[WARPS_PER_BLOCK * TASKS_PER_WARP * 2];

	/* Initialize input matrices to zero */
//	for (int i = threadIdx.x; i < 2 * WARPS_PER_BLOCK * SHMEM_BLOCK_SIZE; i += blockDim.x) {
//		shmem[i] = 0;
//	}
//	__syncthreads();

	for (int C_block = blockIdx.x * WARPS_PER_BLOCK + warp_id; C_block < size; C_block += gridDim.x * WARPS_PER_BLOCK) {

		if (C_block == 0) {
			cur_task = 0;
		} else {
			cur_task = first_task[C_block - 1];
		}

		last_task = first_task[C_block];

		wmma::fragment<wmma::matrix_a, 16, 16, 16, valueIn, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, valueIn, wmma::row_major> b_frag;
		wmma::fragment<wmma::accumulator, 16, 16, 16, valueOut> c_frag;
		wmma::fill_fragment(c_frag, 0.0f);

		for (; cur_task < last_task; cur_task += TASKS_PER_WARP) {
			int pos = cur_task * 2 + lane_id;
			if (pos < last_task * 2) {
				uint64_t task = task_list[pos];
				if (lane_id % 2 == 0) {
					bmps[lane_id] = A_bmps[task];
					first_values[lane_id] = first_values_A[task];
				} else {
					bmps[lane_id] = B_bmps[task];
					first_values[lane_id] = first_values_B[task];
				}
			}

			for (int i = 0; (i < TASKS_PER_WARP * 2) && (cur_task + i/2 < last_task); i += 2) {
				wmma::fill_fragment(a_frag, 0.0f);
				wmma::fill_fragment(b_frag, 0.0f);

				uint64_t bmp_A = bmps[i];
				uint64_t bmp_B = bmps[i + 1];

				valueIn *A_block_values = A_values + first_values[i];
				valueIn *B_block_values = B_values + first_values[i + 1];
				/* Cargo en memoria compartida los valores */
				uint64_t my_bit = bmp_A & (1ULL << BLOCK_SIZE - 1 - frag_start);
				if (my_bit) {
					int pos = __popcll(bmp_A >> (BLOCK_SIZE - frag_start));
					a_frag.x[0] = A_block_values[pos];//ld_gbl_cg(values + pos);
				}
				my_bit = bmp_A & (1ULL << BLOCK_SIZE - 1 - (frag_start + 1));
				if (my_bit) {
					int pos = __popcll(bmp_A >> (BLOCK_SIZE - (frag_start + 1)));
					a_frag.x[1] = A_block_values[pos];//ld_gbl_cg(values + pos);
				}

				my_bit = bmp_B & (1ULL << BLOCK_SIZE - 1 - frag_start_t);
				if (my_bit) {
					int pos = __popcll(bmp_B >> (BLOCK_SIZE - frag_start_t));
					b_frag.x[0] = B_block_values[pos];//ld_gbl_cg(values + pos);
				}
				my_bit = bmp_B & (1ULL << BLOCK_SIZE - 1 - (frag_start_t + 1));
				if (my_bit) {
					int pos = __popcll(bmp_B >> (BLOCK_SIZE - (frag_start_t + 1)));
					b_frag.x[1] = B_block_values[pos];//ld_gbl_cg(values + pos);
				}

				wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

			}
		}

//		if (blockIdx.x == 0 && threadIdx.x == 0) {
//			printf("\n");
//			for (int i = 0; i < 16 * 16; i++) {
//				if (i % 16 == 0) {
//					printf("\n");
//				}
//				printf("%f ", (float)out[i]);
//			}
//		}
//		__syncwarp();

		if (C_bmps[C_block] & (uint64_t(1) << BLOCK_SIZE - 1 - frag_start)) {
			int pos = __popcll(C_bmps[C_block] >> (BLOCK_SIZE - frag_start));
			C_values[offsets[C_block] + pos] =
							c_frag.x[0];
		}
		if (C_bmps[C_block] & (uint64_t(1) << BLOCK_SIZE - 1 - (frag_start + 1))) {
			int pos = __popcll(C_bmps[C_block] >> (BLOCK_SIZE - (frag_start + 1)));
			C_values[offsets[C_block] + pos] =
							c_frag.x[1];
		}

	}

}


//Tensor naive
template <class valueIn, class valueOut>
__global__
void multiplyV11(uint64_t* task_list, uint64_t *first_task,
		valueIn *A_values, valueIn *B_values, valueOut *C_values, uint64_t *first_values_A, uint64_t *first_values_B,
              uint64_t *A_bmps, uint64_t *B_bmps, uint64_t *C_bmps, uint64_t *offsets, uint64_t size// uint64_t *A_keys, uint64_t *C_keys
              ) {

	extern __shared__  valueIn shmem[];
	uint64_t cur_task, last_task;


	const int lane_id = threadIdx.x % 32;
	const int warp_id = threadIdx.x / 32;
	const int row = (lane_id / 8);
	const int col = lane_id % 8;
	const int t_lane_id = col * 8 + row;
	const int id = row * SHMEM_BLOCK_DIM + col;

	valueIn *block_A = &shmem[warp_id * SHMEM_BLOCK_SIZE];
	valueIn *block_B = &block_A[WARPS_PER_BLOCK * SHMEM_BLOCK_SIZE];
	valueOut *out = (valueOut *)&shmem[2 * WARPS_PER_BLOCK * SHMEM_BLOCK_SIZE] + warp_id * SHMEM_BLOCK_SIZE;
	uint64_t *bmps = (uint64_t*)((valueOut *)&shmem[2 * WARPS_PER_BLOCK * SHMEM_BLOCK_SIZE] + WARPS_PER_BLOCK * SHMEM_BLOCK_SIZE) + warp_id * TASKS_PER_WARP * 2;
	uint64_t *first_values = &bmps[WARPS_PER_BLOCK * TASKS_PER_WARP * 2];

	/* Initialize input matrices to zero */
	for (int i = threadIdx.x; i < 2 * WARPS_PER_BLOCK * SHMEM_BLOCK_SIZE; i += blockDim.x) {
		shmem[i] = 0;
	}
	__syncthreads();

	for (int C_block = blockIdx.x * WARPS_PER_BLOCK + warp_id; C_block < size; C_block += gridDim.x * WARPS_PER_BLOCK) {

		if (C_block == 0) {
			cur_task = 0;
		} else {
			cur_task = first_task[C_block - 1];
		}

		last_task = first_task[C_block];

		wmma::fragment<wmma::matrix_a, 16, 16, 16, valueIn, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, valueIn, wmma::row_major> b_frag;
		wmma::fragment<wmma::accumulator, 16, 16, 16, valueOut> c_frag;
		wmma::fill_fragment(c_frag, 0.0f);

		for (; cur_task < last_task; cur_task += TASKS_PER_WARP) {
			int pos = cur_task * 2 + lane_id;
			if (pos < last_task * 2) {
				uint64_t task = task_list[pos];
				if (lane_id % 2 == 0) {
					bmps[lane_id] = A_bmps[task];
					first_values[lane_id] = first_values_A[task];
				} else {
					bmps[lane_id] = B_bmps[task];
					first_values[lane_id] = first_values_B[task];
				}
			}

			__syncwarp();

			for (int i = 0; (i < TASKS_PER_WARP * 2) && (cur_task + i/2 < last_task); i += 2) {
				uint64_t bmp_A = bmps[i];
				uint64_t bmp_B = bmps[i + 1];

				valueIn *A_block_values = A_values + first_values[i];
				valueIn *B_block_values = B_values + first_values[i + 1];
				/* Cargo en memoria compartida los valores */
				__syncwarp();
				shmem_load<valueIn>(bmp_A, block_A, A_block_values, id, lane_id);
				shmem_load<valueIn>(bmp_A, block_A + (BLOCK_HEIGHT / 2) * SHMEM_BLOCK_DIM, A_block_values, id, lane_id + WARP_SIZE);
				shmem_load<valueIn>(bmp_B, block_B, B_block_values, id, t_lane_id);
				shmem_load<valueIn>(bmp_B, block_B + (BLOCK_HEIGHT / 2) * SHMEM_BLOCK_DIM, B_block_values, id, t_lane_id + BLOCK_WIDTH / 2);
				__syncwarp();

				wmma::load_matrix_sync(a_frag, block_A, 16);
				wmma::load_matrix_sync(b_frag, block_B, 16);
				wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

			}
		}
		wmma::store_matrix_sync(out, c_frag, 16, wmma::mem_row_major);


//		if (blockIdx.x == 1 && threadIdx.x == 32) {
//			printf("\n");
//			for (int i = 0; i < 16 * 16; i++) {
//				if (i % 16 == 0) {
//					printf("\n");
//				}
//				printf("%f ", (float)out[i]);
//			}
//		}
//		__syncwarp();

		if (C_bmps[C_block] & (uint64_t(1) << BLOCK_SIZE - 1 - lane_id)) {
			int pos = __popcll(C_bmps[C_block] >> (BLOCK_SIZE - lane_id));
			C_values[offsets[C_block] + pos] =
							out[id];
		}
		if (C_bmps[C_block] & (uint64_t(1) << BLOCK_SIZE - 1 - (lane_id + WARP_SIZE))) {
			int pos = __popcll(C_bmps[C_block] >> (BLOCK_SIZE - (lane_id + WARP_SIZE)));
			C_values[offsets[C_block] + pos] =
							out[id + (BLOCK_HEIGHT / 2) * SHMEM_BLOCK_DIM];
		}

	}

}

void cusparse_spmv(cusp::csr_matrix<int, float, cusp::device_memory> &A, float* v, float* u);

void cusparse_multiply(cusp::csr_matrix<int, float,cusp::device_memory> &A,
		cusp::csr_matrix<int, float,cusp::device_memory> &B, cusp::csr_matrix<int, float,cusp::device_memory> &res);
void cusparse_multiply(cusp::csr_matrix<int, double,cusp::device_memory> &A,
		cusp::csr_matrix<int, double,cusp::device_memory> &B, cusp::csr_matrix<int, double,cusp::device_memory> &res);

struct multiplication_checker: public thrust::unary_function<task_list_elem, bool> {
	__device__

	bool operator()(task_list_elem task) {
		uint64_t first_bmp = A_bmps[task.first];
		uint64_t second_bmp = B_bmps[task.second];
		for (int i = 0; i < 8; i++) {
			uint64_t first_bmp_row = (first_bmp << (8 * i)) & 0xFF00000000000000;
			for (int j = 0; j < 8; j++) {
	//				for (int k = 0; k < 8; k++) {
					if (first_bmp_row & (second_bmp << (8 * j)) & 0xFF00000000000000 ) {
						return false;
					}
	//					if (((first_bmp & (uint64_t(1) << (63 - (i * 8 + k)))) && (second_bmp & (uint64_t(1) << (63 - (j * 8 + k)))))) {
	//						res |= uint64_t(1) << (63 - (i * 8 + j));
	//					}
	//				}
			}
		}
		return true;
	}

//	bool operator()(task_list_elem task) {
//		uint64_t first_bmp = A_bmps[task.first];
//		uint64_t second_bmp = B_bmps[task.second];
//		uint64_t second_bmp_transposed;
//		/* Se transponen los bits del segundo bitmap */
//		for (int i = 0; i < 8; i++) {
//			*((unsigned char *)&second_bmp_transposed + i) = *((unsigned char *)&second_bmp + 7 - i);
//		}
//		second_bmp = second_bmp_transposed;
//
//		transpose8((unsigned char *)&second_bmp, 1, 1, (unsigned char *)&second_bmp_transposed);
//
//		for (int i = 0; i < 8; i++) {
//			unsigned char &first_B_column = *((unsigned char *)&second_bmp_transposed + 7 - i);
//			for (int j = 0; j < 8; j++) {
//				if (*((unsigned char *)&first_bmp + j) & first_B_column) {
//					return false;
//				}
//			}
//		}
//
//		return true;
//	};

	__device__
	void transpose8(unsigned char A[8], int m, int n,
	                unsigned char B[8]) {
	   unsigned x, y, t;

	   // Load the array and pack it into x and y.

	   x = (A[0]<<24)   | (A[m]<<16)   | (A[2*m]<<8) | A[3*m];
	   y = (A[4*m]<<24) | (A[5*m]<<16) | (A[6*m]<<8) | A[7*m];

	   t = (x ^ (x >> 7)) & 0x00AA00AA;  x = x ^ t ^ (t << 7);
	   t = (y ^ (y >> 7)) & 0x00AA00AA;  y = y ^ t ^ (t << 7);

	   t = (x ^ (x >>14)) & 0x0000CCCC;  x = x ^ t ^ (t <<14);
	   t = (y ^ (y >>14)) & 0x0000CCCC;  y = y ^ t ^ (t <<14);

	   t = (x & 0xF0F0F0F0) | ((y >> 4) & 0x0F0F0F0F);
	   y = ((x << 4) & 0xF0F0F0F0) | (y & 0x0F0F0F0F);
	   x = t;

	   B[0]=x>>24;    B[n]=x>>16;    B[2*n]=x>>8;  B[3*n]=x;
	   B[4*n]=y>>24;  B[5*n]=y>>16;  B[6*n]=y>>8;  B[7*n]=y;
	}

	uint64_t *A_bmps;
	uint64_t *B_bmps;
};

struct bmp_calculator: public thrust::unary_function<task_list_elem, uint64_t> {

	__device__
	uint64_t operator()(task_list_elem task) {
		uint64_t first_bmp = A_bmps[task.first];
		uint64_t second_bmp = B_bmps[task.second];
		uint64_t res = 0;
		for (int i = 0; i < 8; i++) {
			uint64_t first_bmp_row = (first_bmp << (8 * i)) & 0xFF00000000000000;
			uint64_t row_position = 0x8000000000000000 >> i * 8;
			for (int j = 0; j < 8; j++) {
				const uint64_t second_bmp_col = (second_bmp << (8 * j)) & 0xFF00000000000000;
					if (first_bmp_row & second_bmp_col) {
						res |= row_position >> j;
					}
//					if (((first_bmp & (uint64_t(1) << (63 - (i * 8 + k)))) && (second_bmp & (uint64_t(1) << (63 - (j * 8 + k)))))) {
//						res |= uint64_t(1) << (63 - (i * 8 + j));
//					}
//				}
			}
		}

		return res;
	};

	uint64_t *A_bmps;
	uint64_t *B_bmps;
};

struct bmp_sum: public thrust::binary_function<uint64_t, uint64_t, uint64_t> {

	__device__
	uint64_t operator()(uint64_t fst, uint64_t snd) {
		return fst | snd;
	};
};

struct bmp_count: public thrust::unary_function<uint64_t, uint64_t> {
	__device__
	uint64_t operator()(uint64_t bmp) {
		return __popcll(bmp);
	};
};

template <class valueIn, class valueOut>
inline void bmSparse_mult(bmSpMatrix<valueIn> &A, bmSpMatrix<valueIn> &B, bmSpMatrix<valueOut> &C, bool mode, bool VERBOSE, long tc_version) {
	
	std::chrono::time_point<std::chrono::steady_clock> start, end;
       
	auto startFunc = std::chrono::steady_clock::now();
	int parsingTime, bmp_reduction_size;

	if (VERBOSE) {
		start = std::chrono::steady_clock::now();
	}

	// First step: for every row in B', count the number of non-empty blocks
	thrust::constant_iterator<uint64_t> ones_it(1);
	thrust::device_vector<uint64_t> B_blocks_per_row(B.block_num);
//	cudaProfilerStart();
	thrust::reduce_by_key(B.keys.begin(),
			B.keys.end(),
			ones_it,
			thrust::make_discard_iterator(),
			B_blocks_per_row.begin(),
			is_same_row());
//	cudaProfilerStop();

	if (VERBOSE) {
		end = std::chrono::steady_clock::now();
		parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		std::cout << "T_1: " << parsingTime << " μs \n";
		start = std::chrono::steady_clock::now();
	}
	
	
	// Second step: for each block in A, gather the B' count associated with it 

	thrust::device_vector<uint64_t> gather(A.keys.size());
	thrust::device_vector<uint64_t> A_columns(A.keys.size());
	thrust::transform(A.keys.begin(), A.keys.end(), A_columns.begin(),
			key_to_col()); // in-place transformation
 	thrust::gather(A_columns.begin(), A_columns.end(), B_blocks_per_row.begin(),
			gather.begin());

	if (VERBOSE) {
		end = std::chrono::steady_clock::now();
		parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		std::cout << "T_2: " << parsingTime << " μs \n";
		start = std::chrono::steady_clock::now();
	}
	


	// Third step, part 1: generate pos vector 

	thrust::device_vector<uint64_t> pos = thrust::device_vector<uint64_t>(B_blocks_per_row.size());
	thrust::exclusive_scan(B_blocks_per_row.begin(), B_blocks_per_row.end(), pos.begin());
	

	
	// Third step, part 2: expand 
	thrust::device_vector<uint64_t> first_positions(A.keys.size());
	thrust::exclusive_scan(gather.begin(), gather.end(),
			first_positions.begin());

  

	// task_list is zipped with exp_a_keys 
	uint64_t task_list_size = first_positions.back() + gather.back();
	thrust::device_vector<task_list_elem> task_list(task_list_size);
        
        

	if (VERBOSE) {
		bmp_reduction_size = task_list_size;
		std::cout << "Task list size: " << task_list_size << std::endl;
	}

	thrust::constant_iterator<uint64_t> ones(1);
	
	
	
//	offsets_from_positions task_creator_functor;
//	task_creator_functor.A_keys = thrust::raw_pointer_cast(A.keys.data());
//	task_creator_functor.pos = thrust::raw_pointer_cast(pos.data());
//	auto initial_task_elems = thrust::make_transform_iterator(thrust::counting_iterator<uint64_t>(0), task_creator_functor);

	thrust::device_vector<uint64_t> task_keys(task_list_size);
	thrust::counting_iterator<uint64_t> iter(0);
	thrust::scatter(iter,
			iter + A.keys.size(),
			first_positions.begin(),
			task_keys.begin()
			);
	thrust::maximum<uint64_t> max_op;
	thrust::inclusive_scan(task_keys.begin(),
			task_keys.begin() + task_list_size,
			task_keys.begin(),
			max_op
			);

	thrust::device_vector<uint32_t> idx(task_list_size);
	thrust::exclusive_scan_by_key(task_keys.begin(),
			task_keys.end(),
			ones,
			idx.begin()
			);

	task_creator task_creator_functor;
	task_creator_functor.A_keys = thrust::raw_pointer_cast(A.keys.data());
	task_creator_functor.pos = thrust::raw_pointer_cast(pos.data());

	thrust::transform(task_keys.begin(),
			task_keys.end(),
			idx.begin(),
			task_list.begin(),
			task_creator_functor
			);

	if (VERBOSE) {
		end = std::chrono::steady_clock::now();
		parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		std::cout << "T_3: " << parsingTime << " μs \n";
		start = std::chrono::steady_clock::now();
	}
 	
	// Fourth step: 
	// 4.1 Remove and sort 

	multiplication_checker checker;
	checker.A_bmps = thrust::raw_pointer_cast(A.bmps.data());
	checker.B_bmps = thrust::raw_pointer_cast(B.bmps.data());
	auto task_list_end = thrust::remove_if(thrust::device, task_list.begin(), task_list.begin() + task_list_size, checker);
	task_list_size = task_list_end - task_list.begin();


	if (VERBOSE) {
		bmp_reduction_size -= task_list_size;
		std::cout << "Bmp reduction: " << bmp_reduction_size << std::endl;
	 	end = std::chrono::steady_clock::now();
		parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		std::cout << "T_4: " << parsingTime << " μs " << std::endl;
		start = std::chrono::steady_clock::now();
	}

	// Fourth step: 
	// 4.1 Sort and calculate bmps 
	
	if(mode == 2 || (mode == 0 && task_list_size < BORDER)){
                //std::cout << "Traditional sort - ";
                is_less_ik sort_functor;
                sort_functor.A_keys = thrust::raw_pointer_cast(A.keys.data());
                sort_functor.B_keys = thrust::raw_pointer_cast(B.keys.data());
                thrust::sort(task_list.begin(),
                                task_list.begin() + task_list_size,
                                sort_functor);
        }else if(mode == 1 || mode == 0){

                thrust::device_vector<int> i_task_list(task_list_size);
                uint64_t *A_keys, *B_keys;
                A_keys = thrust::raw_pointer_cast(A.keys.data());
                B_keys = thrust::raw_pointer_cast(B.keys.data());

                auto getIK = [=]  __device__ (task_list_elem x) {return (A_keys[x.first] & 0xFFFFFFFF00000000) | (B_keys[x.second] & 0x00000000FFFFFFFF);};
                //auto getI = [=] __device__ (uint64_t x) {return (x & 0xFFFFFFFF00000000) >> 32;};
                auto getI = [=] __device__ (task_list_elem x) {return (A_keys[x.first] & 0xFFFFFFFF00000000) >> 32;};

                thrust::transform(task_list.begin(), task_list.begin() + task_list_size,
                                i_task_list.begin(),
                                getI
                );

                auto end_i = thrust::reduce_by_key(thrust::device,
                                                i_task_list.begin(),
                                                i_task_list.begin() + task_list_size,
                                                ones_it,
                                                task_keys.begin(),
                                                i_task_list.begin());
                
                thrust::transform(task_list.begin(), task_list.end(),
                                task_keys.begin(),
                                getIK
                );

                thrust::exclusive_scan(i_task_list.begin(),
                                end_i.second +1,
                                i_task_list.begin()
                );

                int m = std::distance(i_task_list.begin(),end_i.second + 1);

                uint64_t    *key_d = thrust::raw_pointer_cast(&task_keys[0]);
                task_list_elem *val_d = thrust::raw_pointer_cast(&task_list[0]);
                int *seg_d = thrust::raw_pointer_cast(&i_task_list[0]);
                auto startI = std::chrono::steady_clock::now();
                bb_segsort(key_d, val_d, task_list_size, seg_d, m);
                if(VERBOSE){
                        auto endI = std::chrono::steady_clock::now();
                        parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(endI - startI).count();
                        std::cout << "Segmented sort: " << parsingTime << " μs \n";
                }
        }




	if(VERBOSE) {
		end = std::chrono::steady_clock::now();
		parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		std::cout << "T_5: " << parsingTime << " μs \n";
		start = std::chrono::steady_clock::now();
	}


	// 4.2 Se determina el layout de C 

	thrust::device_vector<task_list_elem> iter_keys(task_list_size);
	thrust::plus<uint64_t> plus;
	// El siguiente vector tendrá, en el i-esimo elemento,
	 //el indice del primer elemento del task list que le toca al bloque i 
	//thrust::device_vector<uint64_t> first_task(task_list_size);

	is_same_ik count_functor;
	count_functor.A_keys = thrust::raw_pointer_cast(A.keys.data());
	count_functor.B_keys = thrust::raw_pointer_cast(B.keys.data());
	auto new_end_2 = thrust::reduce_by_key(thrust::device,
											task_list.begin(),
											task_list.begin() + task_list_size,
											ones,
											iter_keys.begin(),
											task_keys.begin(),
											count_functor,
											plus);
	auto C_size = new_end_2.first - iter_keys.begin();

	thrust::device_vector<uint64_t> C_keys(C_size);
	task_elem_to_C_key C_key_generator_functor;
	C_key_generator_functor.A_keys = thrust::raw_pointer_cast(A.keys.data());
	C_key_generator_functor.B_keys = thrust::raw_pointer_cast(B.keys.data());
	thrust::transform(iter_keys.begin(), iter_keys.begin() + C_size, C_keys.begin(),
			C_key_generator_functor);
	thrust::inclusive_scan(task_keys.begin(), new_end_2.second,
			task_keys.begin());

	if (VERBOSE) {
		end = std::chrono::steady_clock::now();
		parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		std::cout << "T_6: " << parsingTime << " μs \n";
		start = std::chrono::steady_clock::now();
	}


	bmp_calculator bmp_calculator_func;
	bmp_calculator_func.A_bmps = thrust::raw_pointer_cast(A.bmps.data());
	bmp_calculator_func.B_bmps = thrust::raw_pointer_cast(B.bmps.data());
	auto bmps_from_task = thrust::make_transform_iterator(task_list.begin(), bmp_calculator_func);

	thrust::device_vector<uint64_t> C_bmps(task_list_size);

	is_same_ik task_equality_func;
	task_equality_func.A_keys = thrust::raw_pointer_cast(A.keys.data());
	task_equality_func.B_keys = thrust::raw_pointer_cast(B.keys.data());
	auto new_end_d2 = thrust::reduce_by_key(task_list.begin(),
											task_list.begin() + task_list_size,
											bmps_from_task,
											iter_keys.begin(),
											C_bmps.begin(),
											task_equality_func,
											bmp_sum());

	// Calculate C value offsets and nnz 
	uint64_t last_block_nnz, last_offset;
	thrust::device_vector<uint64_t> offsets(C_size + 1);
	thrust::transform(
			C_bmps.begin(),
			C_bmps.begin() + C_size,
			offsets.begin(),
			bmp_count()
			);

	cudaMemcpy((void*)&last_block_nnz, (uint64_t*)thrust::raw_pointer_cast(offsets.data()) + C_size - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	thrust::exclusive_scan(thrust::device, offsets.begin(), offsets.end(), offsets.begin(), 0);

	if (VERBOSE) {
		end = std::chrono::steady_clock::now();
		parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		std::cout << "T_9: " << parsingTime << " μs \n";
		start = std::chrono::steady_clock::now();
	}

	// Calculate non-zero positions 
	cudaMemcpy((void*)&last_offset, (uint64_t*)thrust::raw_pointer_cast(offsets.data()) + C_size - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	uint64_t nnz = last_block_nnz + last_offset;

	// Preparar parametros y hacer la llamada 
	uint64_t *C_keys_raw = thrust::raw_pointer_cast(C_keys.data());
	uint64_t *task_list_raw = (uint64_t *) thrust::raw_pointer_cast(task_list.data());
	uint64_t *first_task_raw = thrust::raw_pointer_cast(task_keys.data());
	uint64_t *offsets_raw = thrust::raw_pointer_cast(offsets.data());
	valueIn *A_values_raw = thrust::raw_pointer_cast(A.values.data());
	valueIn *B_values_raw = thrust::raw_pointer_cast(B.values.data());
	thrust::device_vector<valueOut> C_values(nnz);
	valueOut *C_values_raw = thrust::raw_pointer_cast(C_values.data());
	uint64_t *first_values_A = thrust::raw_pointer_cast(A.offsets.data());
	uint64_t *first_values_B = thrust::raw_pointer_cast(B.offsets.data());
	uint64_t *A_bmps_raw = thrust::raw_pointer_cast(A.bmps.data());
	uint64_t *B_bmps_raw = thrust::raw_pointer_cast(B.bmps.data());
	uint64_t *C_bmps_raw = thrust::raw_pointer_cast(C_bmps.data());


	start = std::chrono::steady_clock::now();

	uint64_t *A_keys_raw = thrust::raw_pointer_cast(A.keys.data());
	const int input_size = WARPS_PER_BLOCK * 2 * SHMEM_BLOCK_SIZE * sizeof(valueIn);
	const int output_size = WARPS_PER_BLOCK * TC_BLOCK_SIZE * sizeof(valueOut);
	const int parallel_tasks_size = WARPS_PER_BLOCK * TASKS_PER_WARP * 4 * sizeof(uint64_t);

	switch(tc_version) {
	  case 1:
	    multiplyV11<<<C_keys.size() / WARPS_PER_BLOCK, 64, input_size + output_size + parallel_tasks_size>>>(task_list_raw, first_task_raw, A_values_raw, B_values_raw,
			C_values_raw, first_values_A, first_values_B, A_bmps_raw, B_bmps_raw, C_bmps_raw, offsets_raw, C_keys.size());//, A_keys_raw, C_keys_raw);
	    break;
	  case 2:
	    multiplyV12<<<C_keys.size() / 2, 64, parallel_tasks_size>>>(task_list_raw, first_task_raw, A_values_raw, B_values_raw,
			C_values_raw, first_values_A, first_values_B, A_bmps_raw, B_bmps_raw, C_bmps_raw, offsets_raw, C_keys.size());//, A_keys_raw, C_keys_raw);
	    break;
	  case 3:
	    multiplyV13<<<C_keys.size() / (WARPS_PER_BLOCK * PARALLEL_BLOCKS), 64, parallel_tasks_size>>>(task_list_raw, first_task_raw, A_values_raw, B_values_raw,
			C_values_raw, first_values_A, first_values_B, A_bmps_raw, B_bmps_raw, C_bmps_raw, offsets_raw, C_keys.size());//, A_keys_raw, C_keys_raw);
	    break;
	  case 4:
	    multiplyV14<<<C_keys.size() / (WARPS_PER_BLOCK * PARALLEL_BLOCKS), 64, parallel_tasks_size>>>(task_list_raw, first_task_raw, A_values_raw, B_values_raw,
			C_values_raw, first_values_A, first_values_B, A_bmps_raw, B_bmps_raw, C_bmps_raw, offsets_raw, C_keys.size());//, A_keys_raw, C_keys_raw);
	    break;
	  case 5:
	    multiplyV15<<<C_keys.size() / WARPS_PER_BLOCK, CUDA_BLOCK_SIZE, input_size + parallel_tasks_size>>>(task_list_raw, first_task_raw, A_values_raw, B_values_raw,
			C_values_raw, first_values_A, first_values_B, A_bmps_raw, B_bmps_raw, C_bmps_raw, offsets_raw, C_keys.size());//, A_keys_raw, C_keys_raw);
	    break;
	  default:
	    break;
	}

	
	cudaDeviceSynchronize();


	if (VERBOSE) {
		end = std::chrono::steady_clock::now();
		parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		std::cout << "T_7: " << parsingTime << " μs \n";
		start = std::chrono::steady_clock::now();
	}
	


	// Create C bmSpMatrix object 
	C.num_rows = A.num_rows;
	C.num_cols = B.num_cols;
	C.nnz = C_values.size();
	C.block_num = C_size;

	// Swapping avoids memory transfers 
	C.keys.swap(C_keys);
	C.bmps.swap(C_bmps);
	C.offsets.swap(offsets);
	C.values.swap(C_values);


	if(VERBOSE){
		start = std::chrono::steady_clock::now();
	}


	if(FREE_VEC){
		B_blocks_per_row.clear();
		B_blocks_per_row.shrink_to_fit();
                gather.clear();
                gather.shrink_to_fit();
                A_columns.clear();
                A_columns.shrink_to_fit();
                pos.clear();
                pos.shrink_to_fit();
                first_positions.clear();
                first_positions.shrink_to_fit();
                task_list.clear();
                task_list.shrink_to_fit();
                task_keys.clear();
                task_keys.shrink_to_fit();
                idx.clear();
                idx.shrink_to_fit();
		iter_keys.clear();
                iter_keys.shrink_to_fit();
	}

        if (VERBOSE) {
                end = std::chrono::steady_clock::now();
                parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                std::cout << "Free: " << parsingTime << " μs \n";
                start = std::chrono::steady_clock::now();
        }


	
	auto endFunc = start = std::chrono::steady_clock::now();
	parsingTime =std::chrono::duration_cast<std::chrono::microseconds>(endFunc - startFunc).count();
        std::cout << "Toda F: " << parsingTime << " μs \n";

	return;
}


int main(int argc, char** argv) {
	/* Dump CSR and bmSpMatrix if needed */
	bool exec_cusparse = *argv[4] == '1';
	bool exec_bmSparse = *argv[5] == '1';
	bool check_correctness = *argv[6] >= '1';
	bool segmented = true;
	bool VERBOSE = false;
	bool exec_spmv=false;
	long tc_version = 5;
	cudaFree(0);	
	if(argc <7){
		std::cout << "./main MatrixFolder A_Matrix B_Matrix exec_cusparse exec_bmSparse check_correctness verbose segmented" << std::endl;
		return 1;
	}
	if(argc > 7)
		VERBOSE = *argv[7] =='1';
	
	if(argc > 8)
		segmented = *argv[8] == '1'; 

	if(argc > 9)
		tc_version = strtol(argv[9], NULL, 10);

	if(argc > 10)
		exec_spmv = *argv[10] == '1';
		

//		for (int i = 2; i < 4; i++) {
//		std::string path = std::string(argv[1]) + "/" + std::string(argv[i]);
//		ifstream dump_made(path + "_keys");
////		if (!dump_made) {
//			write_dump(argv[1], argv[i]);
////		}
//		dump_made = ifstream(path + "_csr");
//		if (!dump_made && exec_cusparse) {
//			cusp::csr_matrix<int, float, cusp::host_memory> matrix;
//			cusp::io::read_matrix_market_file(matrix, path + ".mtx");
//			cusp::io::write_binary_file(matrix, path + "_csr");
//		}
//	}

//	thrust::host_vector<uint64_t> keys_new = read_vector<uint64_t>("/home/renzo/Desktop/proyecto/data/test/test_keys");
//	thrust::host_vector<uint64_t> bmps_new = read_vector<uint64_t>("/home/renzo/Desktop/proyecto/data/test/test_bmps");
//	thrust::host_vector<uint64_t> offsets_new = read_vector<uint64_t>("/home/renzo/Desktop/proyecto/data/test/test_offsets");
//	thrust::host_vector<float> values_new = read_vector<float >("/home/renzo/Desktop/proyecto/data/test/test_values");
//	thrust::host_vector<uint64_t> keys_old = read_vector<uint64_t>("/home/renzo/Desktop/proyecto/data/test/test_keys_old");
//	thrust::host_vector<uint64_t> bmps_old = read_vector<uint64_t>("/home/renzo/Desktop/proyecto/data/test/test_bmps_old");
//	thrust::host_vector<uint64_t> offsets_old = read_vector<uint64_t>("/home/renzo/Desktop/proyecto/data/test/test_offsets_old");
//	thrust::host_vector<float> values_old = read_vector<float >("/home/renzo/Desktop/proyecto/data/test/test_values_old");
//
//	std::cout << "EEEEH " << keys_new[0] << " -> "<< bmps_new[0] << std::endl;
//
//	for (int i = 0; i < keys_old.size(); i++) {
//		if (keys_old[i] != keys_new[i]) {
//			std::cout << "Keys diff: " << keys_old[i] << " | " << keys_new[i] << std::endl;
//		}
//		if (bmps_old[i] != bmps_new[i]) {
//			std::cout << "bmps diff: " << bmps_old[i] << " | " << bmps_new[i] << std::endl;
//		}
//	}


	std::string A_path = std::string(argv[1]) + "/" + std::string(argv[2]);
	std::string B_path = std::string(argv[1]) + "/" + std::string(argv[3]);

	std::cout << "A matrix: " << A_path << std::endl;
	std::cout << "B matrix: " << B_path << std::endl;
	std::chrono::steady_clock::time_point start, end;

	cusp::csr_matrix<int, OUTPUT_TYPE, cusp::device_memory> correct_result;
<<<<<<< HEAD
	if (!exec_cusparse) { //CAMBIAR
=======
	if (exec_cusparse) {
>>>>>>> 7a2e3211c04a7e2059e8279160648d6e25bce12e
		start = std::chrono::steady_clock::now();
		cusp::csr_matrix<int, OUTPUT_TYPE, cusp::device_memory> A_csr, B_csr;
		cusp::io::read_matrix_market_file(A_csr, A_path + ".mtx");
		cusp::io::read_matrix_market_file(B_csr, B_path + ".mtx");

		end = std::chrono::steady_clock::now();
		auto parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(
				end - start).count();
		std::cout << "Parsing mtx files / Loading matrices from disk CSR: " << parsingTime
				<< " μs" << std::endl;
		cusparse_multiply(A_csr, B_csr, correct_result);
	}

	if (exec_bmSparse) {
		/* Se leen matrices */
		cudaDeviceSynchronize();
		




		

		start = std::chrono::steady_clock::now();
		bmSpMatrix<half> A_bmSp(A_path + ".mtx", false);
		bmSpMatrix<half> B_bmSp(B_path + ".mtx", true);
	//	auto A_bmSp = read_dump(argv[1], argv[2]);
	//	auto B_bmSp = read_dump(argv[1], argv[3]);

		//std::cout << "num rows: " << B_bmSp.num_rows << std::endl;
		//std::cout << "num cols: " << B_bmSp.num_cols << std::endl;

		end = std::chrono::steady_clock::now();
		auto parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(
				end - start).count();
		std::cout << "Parsing mtx files / Loading matrices from disk BMSP: " << parsingTime
				<< " μs" << std::endl;
		cudaDeviceSynchronize();
		start = std::chrono::steady_clock::now();
		
		bmSpMatrix<OUTPUT_TYPE> C;	

		bmSparse_mult(A_bmSp, B_bmSp, C, segmented, VERBOSE, tc_version);
	
                end = std::chrono::steady_clock::now();		
                auto bmSpTime = std::chrono::duration_cast<std::chrono::microseconds>(
                                end - start).count();
		std::cout << "bmSparse execution: " << bmSpTime << " μs" << std::endl;
	

		if (VERBOSE) {
			std::cout << "(Execution time includes overhead from previous writes to stdout. Set VERBOSE to 0 in order to change this.)" << std::endl;
		}
		std::cout << "C blocks: " << C.keys.size() << std::endl;
		std::cout << "C nnz: " << C.nnz << std::endl;
		std::cout << "C nnz cu: " << correct_result.values.size() << std::endl;
		if (check_correctness) {
			auto ret = C.compare(correct_result);
		}
	}

	if(exec_spmv){
                std::cout << "Running SpMV \n" ;
	
                OUTPUT_TYPE *v, *u, *vcpu;
		
<<<<<<< HEAD
		//start = std::chrono::steady_clock::now();
=======
		start = std::chrono::steady_clock::now();
>>>>>>> 7a2e3211c04a7e2059e8279160648d6e25bce12e
	      	bmSpMatrix<OUTPUT_TYPE> A_matrix(A_path/* + ".mtx"*/, false);


		vcpu =(OUTPUT_TYPE *)  malloc(sizeof(OUTPUT_TYPE)* A_matrix.num_cols);
                cudaMalloc((void**)&v, sizeof(OUTPUT_TYPE)* A_matrix.num_cols);
                cudaMalloc((void**)&u, sizeof(OUTPUT_TYPE)* A_matrix.num_rows);
                for(int i=0; i<A_matrix.num_cols;i++){
                        vcpu[i] = 1;
                }
<<<<<<< HEAD

		start = std::chrono::steady_clock::now();

=======
>>>>>>> 7a2e3211c04a7e2059e8279160648d6e25bce12e
                cudaMemcpy(v, vcpu, sizeof(OUTPUT_TYPE)* A_matrix.num_cols,  cudaMemcpyHostToDevice);

	   	end = std::chrono::steady_clock::now();
           	
                auto parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(
                                end - start).count();
                std::cout << "Parsing mtx files / Loading matrix and vectors: " << parsingTime
                                << " μs" << std::endl;

		OUTPUT_TYPE *u_bmsp, *u_cusp;
                u_bmsp =(OUTPUT_TYPE *)  malloc(sizeof(OUTPUT_TYPE)* A_matrix.num_rows);
                u_cusp =(OUTPUT_TYPE *)  malloc(sizeof(OUTPUT_TYPE)* A_matrix.num_rows);

		//Archivo de salidas
		std::ofstream output;
      		output.open ("histogramas.csv", std::ios_base::app);
		//header
		//output << "Matriz,Preproc Cusparse,Mult Cusparse,Free Cusparse,Total Cusparse,Preproc Bmsparse,Kernel Bmsparse,Free Bmsparse,Total Bmsparse\n";
		//output << "Matriz,Kernel Cusp,Preprocesamiento bmSparse,Kernel bmSparse,Standard Deviation\n";
		//output << "Matriz,Pre Cusp,Mult Cusp,Free Cusp,Pre bmsp, Mult bmsp, Free bmsp\n";
		output << std::string(argv[2]) << ",";
		



		/*Cusparse*/
<<<<<<< HEAD
		auto bmSpTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

		if(exec_cusparse){

            		cusp::csr_matrix<int, OUTPUT_TYPE, cusp::device_memory> A_csr;
               		cusp::io::read_matrix_market_file(A_csr, A_path/* + ".mtx"*/);

			start = std::chrono::steady_clock::now();

			cusparseHandle_t handle;
			cusparseStatus_t status = cusparseCreate(&handle); 

			end = std::chrono::steady_clock::now();
			parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
			std::cout << "cuSparse handle: " << parsingTime << " μs\n"; 
	
			//CUSPARSE_CHECK(status);
			start = std::chrono::steady_clock::now();                
	
			//cusparse_spmv(A_csr, v, u, handle);
			cusparse_spmv(A_csr, v, u, handle, output);
			cudaDeviceSynchronize();
			
			end = std::chrono::steady_clock::now();
			bmSpTime = std::chrono::duration_cast<std::chrono::microseconds>(
	                end - start).count();
                	//cudaDeviceSynchronize();
			std::cout << "Cusparse SpMV execution: " << bmSpTime << " μs\n";
			output << std::to_string(bmSpTime) + ","; //Total Time CUSP
			output << "\n";
			//output << ",";
			cudaMemcpy(u_cusp, u, sizeof(OUTPUT_TYPE)* A_matrix.num_rows, cudaMemcpyDeviceToHost);

		}
	
		if(!exec_cusparse){

=======




                cusp::csr_matrix<int, OUTPUT_TYPE, cusp::device_memory> A_csr;
                cusp::io::read_matrix_market_file(A_csr, A_path/* + ".mtx"*/);

		start = std::chrono::steady_clock::now();

		cusparseHandle_t handle;
		cusparseStatus_t status = cusparseCreate(&handle); 

		end = std::chrono::steady_clock::now();
		parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		std::cout << "cuSparse handle: " << parsingTime << " μs\n"; 

		//CUSPARSE_CHECK(status);
		start = std::chrono::steady_clock::now();                

		//cusparse_spmv(A_csr, v, u, handle);
		cusparse_spmv(A_csr, v, u, handle, output);
		cudaDeviceSynchronize();
		
		end = std::chrono::steady_clock::now();
                auto bmSpTime = std::chrono::duration_cast<std::chrono::microseconds>(
                                end - start).count();
                //cudaDeviceSynchronize();
		std::cout << "Cusparse SpMV execution: " << bmSpTime << " μs\n";
		output << std::to_string(bmSpTime) + ","; //Total Time CUSP
		//output << ",";
		cudaMemcpy(u_cusp, u, sizeof(OUTPUT_TYPE)* A_matrix.num_rows, cudaMemcpyDeviceToHost);
	
>>>>>>> 7a2e3211c04a7e2059e8279160648d6e25bce12e
		/*BmSparse*/
               	cudaDeviceSynchronize();
                start = std::chrono::steady_clock::now();           
		//std::cout << "    A matrix rows: " << A_matrix.num_rows << "\n    A matrix columns " << A_matrix.num_cols << "\n";

		//bmSparse_SpMV(A_matrix, v, u);
		bmSparse_SpMV(A_matrix, v, u, output);
		cudaDeviceSynchronize();

		//cudaDeviceSynchronize();
		end = std::chrono::steady_clock::now();
                bmSpTime = std::chrono::duration_cast<std::chrono::microseconds>(
                                end - start).count();
                
		std::cout << "bmSparse SpMV execution: " << bmSpTime << " μs" << std::endl;
		output << std::to_string(bmSpTime) + ","; //Total Time BMSP

		output << "\n";
                cudaMemcpy(u_bmsp, u, sizeof(OUTPUT_TYPE)* A_matrix.num_rows,  cudaMemcpyDeviceToHost);
		
                for(int i=0; i<A_matrix.num_rows; i++){
                	//printf("Valor bmsp: %f, Valor cusp: %f\n", u_bmsp[i], u_cusp[i])       ; 
			float dif = u_bmsp[i] - u_cusp[i];
			if(dif <0) dif = 0-dif;
			if(dif>0.00001){
				printf("Diff: %.16f\n", dif);
				//printf("Value bmsp %i: %.16e.  \n",i, u_bmsp[i]);
				printf("Error in %i. Value bmsp: %.16e. Value cusp: %.16e\n",i,u_bmsp[i], u_cusp[i]);
				break;
			}
        }

<<<<<<< HEAD
}
=======

>>>>>>> 7a2e3211c04a7e2059e8279160648d6e25bce12e
	}

	return 0;
}
