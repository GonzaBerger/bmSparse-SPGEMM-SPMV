/*
 * bmSpMatrix.h
 *
 *  Created on: Apr 15, 2020
 *      Author: renzo
 */

#ifndef BMSPMATRIX_H_
#define BMSPMATRIX_H_

#include <thrust/device_vector.h>
#include <cuda.h>
#include <cusp/coo_matrix.h>

#define BLOCK_WIDTH 8
#define BLOCK_HEIGHT 8
#define BMSP_BLOCK_SIZE (BLOCK_WIDTH * BLOCK_HEIGHT)


template <class valueType>
class bmSpMatrix {
private:
	bool is_same_row(uint64_t key_1, uint64_t key_2);
    cusp::coo_matrix<int,valueType, cusp::host_memory> coo;
	inline void ordered_insertion(thrust::device_vector<uint64_t> &ordered_pos,
			uint64_t new_pos, thrust::device_vector<valueType> &ordered_vals, valueType new_val);
public:
	thrust::device_vector<uint64_t> keys;
	thrust::device_vector<uint64_t> bmps;
	thrust::device_vector<uint64_t> offsets;
	thrust::device_vector<valueType> values;
	int num_rows, num_cols, nnz, block_num;
	bmSpMatrix();
	bmSpMatrix(std::string, bool transpose);
	bmSpMatrix(int num_rows, int num_cols, int block_num, thrust::device_vector<uint64_t> &keys,
			thrust::device_vector<uint64_t> &bmps, thrust::device_vector<uint64_t> &offsets, thrust::device_vector<valueType> &values);
	bool compare(cusp::coo_matrix<int, valueType,cusp::device_memory> matrix);
	void print();
	void generate_coo();
};


#endif /* BMSPMATRIX_H_ */
