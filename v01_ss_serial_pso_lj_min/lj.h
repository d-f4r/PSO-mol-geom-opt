#ifndef LJ_H
#define LJ_H

#include "main.h"

typedef struct {
	double** s_m;
	double** s2_m;
	double** c6_m;
	double** c12_m;
} LJ_Const;

void precalc_lj_const(Atoms* atoms, LJ_Const* lj_arr);
int lj_energy(double* atom_arr, int atomn, LJ_Const* lj_arr, double* energy_out);
int lj_energy_gradient(double* atom_arr, int atomn, LJ_Const* lj_arr, double* energy_gradient_arr_out);

#endif
