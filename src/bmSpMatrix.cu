
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include "bmSpMatrix.h"
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/host_vector.h>
#include <cusp/print.h>
#include <fstream>
#include <limits>
#include "cuda_fp16.h"
#include <map>

template <typename valueType>
bmSpMatrix<valueType>::bmSpMatrix() {
	num_rows = 0;
	num_cols = 0;
	nnz = 0;
}

template <class valueType>
bmSpMatrix<valueType>::bmSpMatrix(int num_rows, int num_cols, int block_num, thrust::device_vector<uint64_t> &keys,
		thrust::device_vector<uint64_t> &bmps, thrust::device_vector<uint64_t> &offsets, thrust::device_vector<valueType> &values) {
	this->num_rows = num_rows;
	this->num_cols = num_cols;
	this->nnz = values.size();
	this->block_num = block_num;

	/* Swapping avoids memory transfers */
	this->keys.swap(keys);
	this->bmps.swap(bmps);
	this->offsets.swap(offsets);
	this->values.swap(values);
}

template <class valueType>
struct block_order: public thrust::binary_function<thrust::tuple<int, int, valueType>, thrust::tuple<int, int, valueType>, bool> {
	__device__
	bool operator()(thrust::tuple<int, int, valueType> fst, thrust::tuple<int, int, valueType> snd) {
		int fst_block_row = thrust::get<0>(fst) / 8;
		int fst_block_col = thrust::get<1>(fst) / 8;
		int snd_block_row = thrust::get<0>(snd) / 8;
		int snd_block_col = thrust::get<1>(snd) / 8;

		if (fst_block_row < snd_block_row) return true;
		if (fst_block_row > snd_block_row) return false;
		if (fst_block_col < snd_block_col) return true;
		if (fst_block_col == snd_block_col) {
			if (transposed) {
				if (thrust::get<1>(fst) < thrust::get<1>(snd)) return true;
				if (thrust::get<1>(fst) > thrust::get<1>(snd)) return false;
				if (thrust::get<0>(fst) < thrust::get<0>(snd)) return true;
	//				if (thrust::get<1>(fst) > thrust::get<1>(snd)) return true;
	//				if (thrust::get<1>(fst) < thrust::get<1>(snd)) return false;
				//if (thrust::get<0>(fst) < thrust::get<0>(snd)) return true;
			} else {
				if (thrust::get<0>(fst) < thrust::get<0>(snd)) return true;
				if (thrust::get<0>(fst) > thrust::get<0>(snd)) return false;
				if (thrust::get<1>(fst) < thrust::get<1>(snd)) return true;
			}
		}
		return false;
	};
	bool transposed;
};

struct coord_to_key: public thrust::unary_function<thrust::tuple<int, int>, uint64_t> {
	__host__ __device__
	uint64_t operator()(thrust::tuple<int, int> coords) {
		return (uint64_t(thrust::get<0>(coords) / 8) << uint64_t(32))
				| (uint64_t(thrust::get<1>(coords) / 8) & 0x00000000FFFFFFFF);
	}
	;
};

struct coord_to_bmp: public thrust::unary_function<thrust::tuple<int, int>, uint64_t> {
	__host__ __device__
	uint64_t operator()(thrust::tuple<int, int> coords) {
		int rel_i = thrust::get<0>(coords) % 8;
		int rel_j = thrust::get<1>(coords) % 8;
		int pos;
		if (transposed) {
			pos = rel_j * 8 + rel_i;
		} else {
			pos = rel_i * 8 + rel_j;
		}
		uint64_t res = (uint64_t(1) << uint64_t(63 - pos));
		return res;
	}
	;
	bool transposed = false;
};

struct bmp_sum: public thrust::binary_function<uint64_t, uint64_t, uint64_t> {

	__device__
	uint64_t operator()(uint64_t fst, uint64_t snd) {
		return fst | snd;
	};
};

template <class valueType>
bmSpMatrix<valueType>::bmSpMatrix(std::string path, bool transposed) {
	std::string first_line;
	std::ifstream file(path);
	std::getline (file, first_line);
	bool symmetric = false;
	if (first_line.find("symmetric") != std::string::npos) {
		std::cout << "TEST";
	    symmetric = true;
	}

	// Ignore comments headers
	while (file.peek() == '%')
		file.ignore(2048, '\n');

	// Read number of rows and columns
	file >> num_rows >> num_cols >> nnz;

	std::vector<int> rows_h, cols_h;
	std::vector<valueType> values_h;
	// fill the matrix with data

	if (symmetric) {
		int added_nnz = 0;
		for (int l = 0; l < nnz; l++) {
			double data;
			int row, col;
			file >> row >> col >> data;
			rows_h.push_back(row - 1);
			cols_h.push_back(col - 1);
			values_h.push_back((valueType)data);
			if (row != col) {
				rows_h.push_back(col - 1);
				cols_h.push_back(row - 1);
				values_h.push_back((valueType)data);
				added_nnz++;
			}
		}
		nnz += added_nnz;
	} else {
		for (int l = 0; l < nnz; l++) {
			double data;
			int row, col;
			file >> row >> col >> data;
			rows_h.push_back(row - 1);
			cols_h.push_back(col - 1);
			values_h.push_back((valueType)data);
		}
	}

	file.close();

	thrust::device_vector<int> rows(rows_h);
	thrust::device_vector<int> cols(cols_h);
	this->values = thrust::device_vector<valueType>(values_h);

	block_order<valueType> block_order_func;
	block_order_func.transposed = transposed;
	auto coo = thrust::make_zip_iterator(thrust::make_tuple(rows.begin(), cols.begin(), values.begin()));
	thrust::sort(coo,
			coo + nnz,
			block_order_func);



	auto coords = thrust::make_zip_iterator(thrust::make_tuple(rows.begin(), cols.begin()));
	thrust::device_vector<uint64_t> coord_to_key_mapping(nnz);
	thrust::transform(coords, coords + nnz, coord_to_key_mapping.begin(), coord_to_key());

	thrust::device_vector<uint64_t> keys(nnz), offsets(nnz);
	/* Calculates keys and offsets vectors */
	thrust::constant_iterator<uint64_t> ones(1);
	auto new_ends = thrust::reduce_by_key(coord_to_key_mapping.begin(),
			coord_to_key_mapping.end(),
			ones,
			keys.begin(),
			offsets.begin()
			);

	thrust::exclusive_scan(thrust::device, offsets.begin(), offsets.end(), offsets.begin(), 0);
//	thrust::host_vector<uint64_t> offsets_deb(offsets.begin(), offsets.end());
	block_num = new_ends.first - keys.begin();
	this->keys = thrust::device_vector<uint64_t>(keys.begin(), keys.begin() + block_num);
	this->offsets = thrust::device_vector<uint64_t>(offsets.begin(), offsets.begin() + block_num);


	/* Calculates bmps vector */
	thrust::device_vector<uint64_t> partial_bmps(nnz);
	this->bmps = thrust::device_vector<uint64_t>(block_num);
	coord_to_bmp coord_to_bmp_func;
	coord_to_bmp_func.transposed = transposed;
	thrust::transform(coords,
			coords + nnz,
			partial_bmps.begin(),
			coord_to_bmp_func
			);

	thrust::equal_to<uint64_t> pred;
	thrust::reduce_by_key(coord_to_key_mapping.begin(),
			coord_to_key_mapping.end(),
			partial_bmps.begin(),
			keys.begin(),
			this->bmps.begin(),
			pred,
			bmp_sum()
			);


}
//
//bmSpMatrix::bmSpMatrix(std::string path) {
//	std::ifstream file(path);
//	// Ignore comments headers
//	while (file.peek() == '%')
//		file.ignore(2048, '\n');
//
//	// Read number of rows and columns
//	file >> num_rows >> num_cols >> nnz;
//	// Create 2D array and fill with zeros
//	std::map<int, std::map<int, uint64_t>> bmSparse;
//	std::map<int, std::map<int, uint64_t>> offsets_map;
//	std::map<int, std::map<int, thrust::device_vector<uint64_t>>>pos_map;
//	std::map<int, std::map<int, thrust::device_vector<valueType >>>value_map;
//
//	// fill the matrix with data
//	for (int l = 0; l < nnz; l++) {
//		valueType test;
//		int row, col;
//		file >> row >> col >> test;
//		row--;
//		col--; // 1-based indices
//		valueType data = test;
//		int block_row = row / 8;
//		int block_col = col / 8;
//		if (block_row == 582 && block_col == 582) {
//			std::cout << row << " " << col << " " << data << std::endl;
//		}
//		uint64_t position = row * num_cols + col;
//		ordered_insertion(pos_map[block_row][block_col], position,
//				value_map[block_row][block_col], data);
//		offsets_map[block_row][block_col] += 1;
//		bmSparse[block_row][block_col] |= uint64_t(1)
//				<< uint64_t(63 - ((row % 8) * 8 + (col % 8)));
//	}
//
//	file.close();
//
//	for (auto const& block_row : bmSparse) {
//		for (auto const& block_col : block_row.second) {
//			uint64_t key = (uint64_t(block_row.first) << uint64_t(32))
//					| uint64_t(block_col.first);
//			keys.push_back(key);
//			bmps.push_back(block_col.second);
//			offsets.push_back(offsets_map[block_row.first][block_col.first]);
//			values.insert(values.end(),
//					value_map[block_row.first][block_col.first].begin(),
//					value_map[block_row.first][block_col.first].end());
//		}
//	}
//
//	thrust::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());
//}

template <class valueType>
inline void bmSpMatrix<valueType>::ordered_insertion(thrust::device_vector<uint64_t> &ordered_pos,
		uint64_t new_pos, thrust::device_vector<valueType> &ordered_vals, valueType new_val) {
	int i = 0;
	for (; i < ordered_pos.size(); i++) {
		if (new_pos < ordered_pos[i]) {
			break;
		}
	}
	ordered_pos.insert(ordered_pos.begin() + i, new_pos);
	ordered_vals.insert(ordered_vals.begin() + i, new_val);
}

template <class valueType>
struct row_order: public thrust::binary_function<
		thrust::tuple<int, int, valueType>, thrust::tuple<int, int, valueType>, bool> {
	__host__ __device__
	bool operator()(thrust::tuple<int, int, valueType>fst,
			thrust::tuple<int, int, valueType> snd) {
		int fst_row = thrust::get<0>(fst);
		int fst_col = thrust::get<1>(fst);
		int snd_row = thrust::get<0>(snd);
		int snd_col = thrust::get<1>(snd);
		return fst_row < snd_row || (fst_row == snd_row && fst_col < snd_col);
	}
	;
};

template <class valueType>
bool bmSpMatrix<valueType>::is_same_row(uint64_t key_1, uint64_t key_2){
	return (key_1 >> 32) == (key_2 >> 32);
}

int get_row(uint64_t key_1){
	return (key_1 >> 32);
}

//template <class valueType>
//void bmSpMatrix<valueType>::print() {
//	if (coo.num_entries == 0) {
//		bmSpMatrix<valueType>::generate_coo();
//	}
//	cusp::print(coo);
//
//}

template <class valueType>
void bmSpMatrix<valueType>::generate_coo() {
	coo.num_rows = num_rows;
    coo.num_cols = num_cols;
    coo.num_entries = nnz;
    thrust::host_vector<uint64_t> h_keys = keys;
    thrust::host_vector<uint64_t> h_bmps = bmps;
    thrust::host_vector<valueType> h_values = values;

    thrust::host_vector<uint64_t> coo_rows, coo_cols;
    thrust::host_vector<valueType> coo_values;
    auto cur_key = h_keys.begin();;
    auto cur_value = h_values.begin();

    for (auto cur_key = h_keys.begin(); cur_key != h_keys.begin() + block_num; cur_key++) {
    	uint64_t block_row = *cur_key >> 32;
    	uint64_t block_col = (*cur_key << 32) >> 32;
    	uint64_t bmp = *(h_bmps.begin() + (cur_key - h_keys.begin()));
    	for (int i = 0; i < BMSP_BLOCK_SIZE; i++) {
    		auto match = bmp & (uint64_t(1) << BMSP_BLOCK_SIZE - 1 - i);
    		int key_num = cur_key - h_keys.begin(); //debug
    		if (match) {
    			auto row = block_row * BLOCK_HEIGHT + (i / BLOCK_WIDTH);
    			auto col = block_col * BLOCK_WIDTH + (i % BLOCK_HEIGHT);
//    			if (row == 8062 && col == 33646) {
//    				std::cout << "block: (" << block_row << ", " << block_col << ")" << std::endl;
//    				std::cout << "key num: " << cur_key - h_keys.begin() << std::endl;
//    			}
    			coo_rows.push_back(row);
    			coo_cols.push_back(col);
    			coo_values.push_back(*cur_value);
    			cur_value++;
    		}
    	}
    }
    auto zip_it = thrust::make_zip_iterator(thrust::make_tuple(coo_rows.begin(), coo_cols.begin(), coo_values.begin()));
	thrust::sort(zip_it, zip_it + nnz, row_order<valueType>());

    for (auto cur_it = zip_it; cur_it != zip_it + nnz; cur_it++) {
    	coo.row_indices.push_back(thrust::get<0>(*cur_it));
    	coo.column_indices.push_back(thrust::get<1>(*cur_it));
    	coo.values.push_back(thrust::get<2>(*cur_it));
    }
}

template <class valueType>
struct delete_zeros: public thrust::unary_function<thrust::tuple<valueType, valueType, valueType>, bool> {
	__device__
	bool operator()(thrust::tuple<valueType, valueType, valueType> matrix_elem) {
		return thrust::get<2>(matrix_elem) == valueType(0);
	};
};

template <class valueType>
struct debug_remove: public thrust::unary_function<thrust::tuple<uint64_t, uint64_t, valueType>, bool> {
	__device__
	bool operator()(thrust::tuple<uint64_t, uint64_t, valueType> elem) {
		return thrust::get<2>(elem) == valueType(0);
	}
};

template <class valueType>
bool bmSpMatrix<valueType>::compare(cusp::coo_matrix<int, valueType,cusp::device_memory> d_matrix) {

	if (coo.num_entries == 0) {
		bmSpMatrix<valueType>::generate_coo();
	}

	auto zipped_matrix = thrust::make_zip_iterator(thrust::make_tuple(
													d_matrix.row_indices.begin(),
													d_matrix.column_indices.begin(),
													d_matrix.values.begin()
													));

	thrust::sort(zipped_matrix, zipped_matrix + d_matrix.num_entries, row_order<valueType>());


	cusp::coo_matrix<int, valueType,cusp::host_memory> matrix = d_matrix;
	bool equal = true;
	int offset = 0;
	int i = 0;
	double count = 0;
	int k = 0;
	double epsilon = 0.00000001;
	for (; i < coo.values.size(); i++) {
		while (matrix.row_indices[i + offset] != coo.row_indices[i]
				 || matrix.column_indices[i + offset] != coo.column_indices[i]) {
			offset++;
		}
		double expected_value = matrix.values[i + offset];;
		if (abs((double)matrix.values[i + offset]) < epsilon) {
			expected_value = 0;
		}
		double real_value = coo.values[i];
		if (abs((double)coo.values[i]) < epsilon) {
			real_value = 0;
		}

		double sum = fabs(expected_value - real_value) / max(fabs(expected_value), epsilon);
		count += sum;
		if (sum > 1000 && k < 1) {
			std::cout << "sum: " << sum << " -> (" << matrix.row_indices[i + offset] << ", " << matrix.column_indices[i + offset] <<  ") -> Expected: " << (double)matrix.values[i + offset] << " | Real: " << (double)coo.values[i] << std::endl;
			std::cout << "exp: " << expected_value << std::endl;
			std::cout << "real: " << real_value << std::endl;
			std::cout << "diff: " << fabs(expected_value - real_value) << std::endl;
			std::cout << "summ calculated: " << fabs(expected_value - real_value) / fabs(expected_value) << std::endl;
			k++;
		}
	}
	std::cout << "Final: " << count / nnz;
	return (equal);

}


template class bmSpMatrix<float>;
template class bmSpMatrix<half>;
template class bmSpMatrix<double>;
