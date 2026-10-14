#ifndef UNIVERSAL_H
#define UNIVERSAL_H

#define WIDTH 7
#define PREC 3
#define COL_PRINT_CUTOFF 10

double** matrix_calloc(int r, int c);
double** matrix_malloc(int r, int c);
void matrix_print(char* title_string, double** matrix, int r, int c);
void drandm(double** a, int n, int m);
void matrix_elemwise_prod(double** a, double** b, double** c,  int m, int n);

#endif
