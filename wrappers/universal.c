#include <stdio.h>
#include <stdlib.h>
#include "universal.h"


double** matrix_calloc(int r, int c) {
	// memory block pou einai prwta ta pointers sta data kai meta ta data
	// meta ka8ena apo ta pointers pou einai sthn arxh deixnoun se ka8e seira
	// sto telos xreiazetai mono ena free()
	// MPORW NA XRHSIMOPOIHSW STANDARD 2D ARRAY INDEXING
	size_t bytes_to_calloc = r * sizeof(double*) + r * c * sizeof(double);
	double** m = calloc(1, bytes_to_calloc);
	if (m == NULL) exit(1);
	double* m_data = (double*)(m + r);
	for (int i = 0; i < r; i++)
		m[i] = m_data + (i * c);

	return m;
}

double** matrix_malloc(int r, int c) {
	size_t bytes_to_calloc = r * sizeof(double*) + r * c * sizeof(double);
	double** m = malloc(bytes_to_calloc);
	if (m == NULL) exit(1);
	double* m_data = (double*)(m + r);
	for (int i = 0; i < r; i++)
		m[i] = m_data + (i * c);

	return m;
}

void drandm(double** a, int m, int n) {
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			a[i][j] = (double)rand() / (double)RAND_MAX;
}

void matrix_print(char* title_string, double** matrix, int r, int c) {
	printf("\n>> %s\n", title_string);
	for (int col_start = 0; col_start < c; col_start += COL_PRINT_CUTOFF) {
		int col_end = (col_start + COL_PRINT_CUTOFF < c) ? col_start + COL_PRINT_CUTOFF : c;
		printf("%-*s", WIDTH, "");
		for (int j = col_start; j < col_end; j++)
			printf(" %-*d", WIDTH, j);
		printf("\n\n");
		for (int i = 0; i < r; i++) {
			printf("%-*d", WIDTH, i);
			for (int j = col_start; j < col_end; j++)
				printf(" %-*.*f", WIDTH, PREC, matrix[i][j]);
			printf("\n");
		}
		printf("\n");
	}
}

void matrix_elemwise_prod(double** a, double** b, double** c,  int m, int n) {
	int total = m * n;
	#pragma omp parallel for if(total >= 1000)
	for (int i = 0; i < total; i++)
		c[0][i] = a[0][i] * b[0][i];
}
