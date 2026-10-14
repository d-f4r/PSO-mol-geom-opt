#ifndef LINALG_TEMP_H
#define LINALG_TEMP_H

void matrix_mul_pointed(double** a, double** b, double** c, int m, int k, int n, double beta);
void matrix_mul_flat(double* a, double* b, double* c, int m, int k, int n, double beta);
void matrix_transpose_pointed(double** a, double** b, int m, int n);
void matrix_transpose_flat(double* a, double* b, int m, int n);
void matrix_svd_flat(double* A, int m, int n, double* S, double* U, double* VT);

#endif
