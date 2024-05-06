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
#define INPUT_TYPE double
#define OUTPUT_TYPE float
#define FREE_VEC 1
#define BORDER 2730000

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

			}
		}

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
					if (first_bmp_row & (second_bmp << (8 * j)) & 0xFF00000000000000 ) {
						return false;
					}
			}
		}
		return true;
	}

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
	thrust::reduce_by_key(B.keys.begin(),
			B.keys.end(),
			ones_it,
			thrust::make_discard_iterator(),
			B_blocks_per_row.begin(),
			is_same_row());

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

	long segmented = 0;
	long tc_version = 5;
	bool VERBOSE = false;

	cudaFree(0);	

	if(argc < 3){
		std::cout << "./main MatrixFolder A_Matrix B_Matrix" << std::endl;
		return 1;
	}

	if(argc > 3)
		segmented = strtol(argv[4], NULL, 10); 

	if(argc > 4)
		tc_version = strtol(argv[5], NULL, 10);

	if(argc > 5)
		VERBOSE = *argv[6] == '1';

		

	std::string A_path = std::string(argv[1]) + "/" + std::string(argv[2]);
	std::string B_path = std::string(argv[1]) + "/" + std::string(argv[3]);

	std::cout << "A matrix: " << A_path << std::endl;
	std::cout << "B matrix: " << B_path << std::endl;
	std::chrono::steady_clock::time_point start, end;

	/* Se leen matrices */

	start = std::chrono::steady_clock::now();
	bmSpMatrix<half> A_bmSp(A_path + ".mtx", false);
	bmSpMatrix<half> B_bmSp(B_path + ".mtx", true);

	end = std::chrono::steady_clock::now();
	auto parsingTime = std::chrono::duration_cast<std::chrono::microseconds>(
			end - start).count();
	std::cout << "Parsing mtx files / Loading matrices from disk BMSP: " << parsingTime
			<< " μs" << std::endl;

	cudaDeviceSynchronize();
	
	bmSpMatrix<OUTPUT_TYPE> C;	

	start = std::chrono::steady_clock::now();

	bmSparse_mult(A_bmSp, B_bmSp, C, segmented, VERBOSE, tc_version);

	end = std::chrono::steady_clock::now();		
	auto bmSpTime = std::chrono::duration_cast<std::chrono::microseconds>(
					end - start).count();
							
	std::cout << "bmSparse execution: " << bmSpTime << " μs" << std::endl;

	std::cout << "C blocks: " << C.keys.size() << std::endl;
	std::cout << "C nnz: " << C.nnz << std::endl;

	return 0;
}
