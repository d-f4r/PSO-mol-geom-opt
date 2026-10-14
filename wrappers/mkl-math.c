#include <mkl.h>
#include "mkl-math.h"


void matrix_mul_pointed(double** a, double** b, double** c, int m, int k, int n, double beta) {
	// C = output arr
	double alpha = 1.0;
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans, CblasNoTrans,
		m, n, k,
		alpha,
		&a[0][0], k,
		&b[0][0], n,
		beta,
		&c[0][0], n
	);
}

void matrix_mul_flat(double* a, double* b, double* c, int m, int k, int n, double beta) {
	double alpha = 1.0;
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans, CblasNoTrans,
		m, n, k,
		alpha,
		&a[0], k,
		&b[0], n,
		beta,
		&c[0], n
	);
}

void matrix_transpose_pointed(double** a, double** b, int m, int n) {
	// B = output arr
	mkl_domatcopy(
		'R', 'T',
		m, n, 1.0,
		&a[0][0], n,
		&b[0][0], m
	);
}

void matrix_transpose_flat(double* a, double* b, int m, int n) {
	mkl_domatcopy(
		'R', 'T',
		m, n, 1.0,
		&a[0], n,
		&b[0], m
	);
}

void matrix_svd_flat(double* A, int m, int n, double* S, double* U, double* VT) {
	// overrides A, V is transposed (V^T)
	char jobu = 'A', jobvt = 'A';
	double work[128];
	int lwork = 128;
	int info = 0;
	dgesvd(&jobu, &jobvt, &m, &n, A, &m, S, U, &m, VT, &n, work, &lwork, &info);
}
