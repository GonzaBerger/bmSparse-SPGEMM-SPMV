/*
 * cuSparse.h
 *
 *  Created on: Apr 13, 2020
 *      Author: renzo
 */

#ifndef CUSPARSE_H_
#define CUSPARSE_H_

#include <cusp/csr_matrix.h>

class CSRMatrix {
public:
	CSRMatrix(std::string);
	CSRMatrix(cusp::csr_matrix<float, float, cusp::host_memory> *matrix);
	CSRMatrix multiply(CSRMatrix matrix);

private:
	cusp::csr_matrix<int,float,cusp::host_memory> matrix_repr;
};

#endif /* CUSPARSE_H_ */
