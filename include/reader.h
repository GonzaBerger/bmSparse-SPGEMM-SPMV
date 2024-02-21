#ifndef READER_HPP_
#define READER_HPP_

#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <cusparse_v2.h>
#include <cuda.h>

typedef thrust::device_vector<uint64_t> uint64_vec;
typedef thrust::device_vector<__half> half_vec;

#define CudaSparseCheck( test ) { CudaSparseCheckCore((test), __FILE__, __LINE__); }

thrust::tuple<int, int, int> mmread_bmSparse(std::string path, uint64_vec &keys,
		uint64_vec &bmps, uint64_vec &offsets, half_vec &values);

#endif /* READER_HPP_ */
