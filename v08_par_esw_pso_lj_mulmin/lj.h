#ifndef LJ_H
#define LJ_H

#include "main.h"

typedef struct {
	double* weights_vec;
	double* e_m;
	double* s_m;
	double* s2_m;
	double* c6_m;
	double* c12_m;
} Constants;

typedef struct {
	double* h; // 3x3
	double* s; // 3
	double* u; // 3x3
	double* vt; // 3x3
	double* r; // 3x3
	double* rotated; // atomn x 3
	double* i; // 3x3 id for check
} Kabsch_Arrays;

typedef enum {
	CALC_ENER = 0,
	CALC_GRAD = 1,
	CALC_HESS = 2
} LJ_Mode;

Constants* precalc_const(Atoms* atoms);

int lennard_jones(
	LJ_Mode mode,
	const double* restrict atom_arr,
	const int atomn,
	const Constants* restrict const_arr,
	double* restrict energy_out,
	double* restrict energy_gradient_arr_out,
	double* restrict grad_norm_sq_out,
	double* restrict hessian_out
);

void calc_com(double* vector_arr, int atomn, double* weights_vec, double mol_weight, double* com_out);
void shift_to(double* vector_arr, double* dest, int dim);
void kabsch_optimal_rot(double* ref_arr, double* arr_to_align, int atomn, Kabsch_Arrays* kabsch_block);
void kabsch_rotation(double* arr_to_rot, int atomn, Kabsch_Arrays* kabsch_block);
void calc_inertia_tens_eigv(double* x_p, int atomn, double* weights_vec, double* eigv_out);
double it_eigv_similarity_check(double* coefs_p1, double* coefs_p2);

#endif
