#ifndef LJ_H
#define LJ_H

#include "main.h"

typedef struct {
	double** e_m;
	double** s_m;
	double** s2_m;
	double** c6_m;
	double** c12_m;
} LJ_Const;

void precalc_lj_const(Atoms* atoms, LJ_Const* lj_arr);
int lj_energy(double* atom_arr, int atomn, LJ_Const* lj_arr, double* energy_gradient_arr, double* energy_out, double* grad_norm_sq_out);
int lj_energy_gradient(double* atom_arr, int atomn, LJ_Const* lj_arr, double* energy_gradient_arr);
double rms_p1_p2(double* a, double* b, int atomn, int uniq_comb);
double p1_p2_norm_sq(double* p1, double* p2, int dim);
int just_lj_energy(double* atom_arr, int atomn, LJ_Const* lj_arr, double* energy_out);
double dist_norm_sq(double* p, int atomn);

#endif
