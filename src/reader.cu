#include <map>
#include <fstream>
#include <bits/stdc++.h>
#include <algorithm>
#include <iostream>
#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include "reader.h"

inline void CudaCheckCore(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"Cuda Error %d : %s %s %d\n", code, cudaGetErrorString(code), file, line);
      exit(code);
   }
}

#define CudaCheck( test ) { CudaCheckCore((test), __FILE__, __LINE__); }
#define CudaCheckAfterCall() { CudaCheckCore((cudaGetLastError()), __FILE__, __LINE__); }

inline void CudaSparseCheckCore(cusparseStatus_t code, const char *file, int line) {
   if (code != CUSPARSE_STATUS_SUCCESS) {
      fprintf(stderr,"Cuda Error %d : %s %s %d\n", code, cusparseGetErrorString(code), file, line);
      exit(code);
   }
}

template <class ObjectType>
ObjectType* allocAndCopy(const ObjectType src[], const int size){
    ObjectType* dest = NULL;
    CudaCheck( cudaMalloc(&dest,size*sizeof(ObjectType)) );
    CudaCheck( cudaMemcpy(dest, src, size*sizeof(ObjectType), cudaMemcpyHostToDevice ) );
    return dest;
}


inline void ordered_insertion(thrust::device_vector<uint64_t> &ordered_pos,
		uint64_t new_pos, thrust::device_vector<__half> &ordered_vals, __half new_val) {
	int i = 0;
	for (; i < ordered_pos.size(); i++) {
		if (new_pos < ordered_pos[i]) {
			break;
		}
	}
	ordered_pos.insert(ordered_pos.begin() + i, new_pos);
	ordered_vals.insert(ordered_vals.begin() + i, new_val);

}

thrust::tuple<int, int, int> mmread_bmSparse(std::string path, uint64_vec &keys,
		uint64_vec &bmps, uint64_vec &offsets, half_vec &values) {

	std::ifstream file(path);
	int num_row, num_col, num_lines;
	// Ignore comments headers
	while (file.peek() == '%')
		file.ignore(2048, '\n');

	// Read number of rows and columns
	file >> num_row >> num_col >> num_lines;
	// Create 2D array and fill with zeros
	std::map<int, std::map<int, uint64_t>> bmSparse;
	std::map<int, std::map<int, uint64_t>> offsets_map;
	std::map<int, std::map<int, thrust::device_vector<uint64_t>>>pos_map;
	std::map<int, std::map<int, thrust::device_vector<__half >>>value_map;

	int count = 0;

	// fill the matrix with data
	for (int l = 0; l < num_lines; l++) {
		float test;
		int row, col;
		file >> row >> col >> test;
		row--;
		col--; // 1-based indices
		__half data = __float2half(test);
		int block_row = row / 8;
		int block_col = col / 8;
		if (bmSparse.count(block_row) == 0
				|| bmSparse[block_row].count(block_col) == 0) {
			count++;
		}
		uint64_t position = (row % 8) * 8 + (col % 8);
		ordered_insertion(pos_map[block_row][block_col], position,
				value_map[block_row][block_col], data);
		offsets_map[block_row][block_col] += 1;
		bmSparse[block_row][block_col] |= uint64_t(1)
				<< uint64_t(63 - ((row % 8) * 8 + (col % 8)));
	}

	file.close();

	//keys[0] = uint64_t(2);
	for (auto const& block_row : bmSparse) {
		for (auto const& block_col : block_row.second) {
			uint64_t key = (uint64_t(block_row.first) << uint64_t(32))
					| uint64_t(block_col.first);
			keys.push_back(key);
			bmps.push_back(block_col.second);
			offsets.push_back(offsets_map[block_row.first][block_col.first]);
			values.insert(values.end(),
					value_map[block_row.first][block_col.first].begin(),
					value_map[block_row.first][block_col.first].end());
		}
	}

	thrust::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());
	//    return make_tuple(keys.begin(),keys.begin(),keys.begin());
	return thrust::make_tuple(num_row, num_col, num_lines);

}

