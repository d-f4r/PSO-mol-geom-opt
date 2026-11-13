#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "universal.h"
#include "lj.h"


void elem_param_match(char* elem, double* sigma, double* epsilon) {
	if (strcmp(elem, "Ar") == 0) {
		*epsilon = 0.997;
		*sigma = 3.4;
	} else {
		fprintf(stderr, "unsupported element, exiting now\n");
		exit(1);
	}
}

void precalc_lj_const(Atoms* atoms, LJ_Const* lj_arr) {
	int atomn = atoms->atomn;
	double epsilon, epsilon1, epsilon2; // kj/mol
	double sigma, sigma1, sigma2; // angstrom

	double** e_m = matrix_calloc(atomn, atomn);
	double** s_m = matrix_calloc(atomn, atomn);
	double** s2_m = matrix_calloc(atomn, atomn);
	double** c6_m = matrix_calloc(atomn, atomn);
	double** c12_m = matrix_calloc(atomn, atomn);

	for (int i = 0; i < atomn; i++) {
		for (int j = i; j < atomn; j++) {
			if (strcmp(atoms->types[i], atoms->types[j]) == 0) {
				elem_param_match(atoms->types[i], &sigma, &epsilon);
			} else {
				elem_param_match(atoms->types[i], &sigma1, &epsilon1);
				elem_param_match(atoms->types[j], &sigma2, &epsilon2);
				sigma = (sigma1 + sigma2) / 2;
				epsilon = sqrt(epsilon1 * epsilon2);
			}
			double sigma_sq = sigma * sigma;
			double sigma_6 = sigma_sq * sigma_sq * sigma_sq;
			double sigma_12 = sigma_6 * sigma_6;

			double c6 = 4 * epsilon * sigma_6;
			double c12 = 4 * epsilon * sigma_12;

			e_m[i][j] = epsilon;
			e_m[j][i] = epsilon;
			s_m[i][j] = sigma;
			s_m[j][i] = sigma;
			s2_m[i][j] = sigma_sq;
			s2_m[j][i] = sigma_sq;
			c6_m[i][j] = c6;
			c6_m[j][i] = c6;
			c12_m[i][j] = c12;
			c12_m[j][i] = c12;

		}
	}
	lj_arr->e_m = e_m;
	lj_arr->s_m = s_m;
	lj_arr->s2_m = s2_m;
	lj_arr->c6_m = c6_m;
	lj_arr->c12_m = c12_m;
}

int lj_energy(double* atom_arr, int atomn, LJ_Const* lj_arr, double* energy_gradient_arr_out, double* energy_out, double* grad_norm_sq_out) {

	memset(energy_gradient_arr_out, 0, 3 * atomn * sizeof(double));

	double energy = 0;
	double grad_norm_sq = 0;
	for (int i = 0; i < atomn; i++) {
		for (int j = i + 1; j < atomn; j++) {
			double dx = (atom_arr[3 * i] - atom_arr[3 * j]);
			double dy = (atom_arr[3 * i + 1] - atom_arr[3 * j + 1]);
			double dz = (atom_arr[3 * i + 2] - atom_arr[3 * j + 2]);
			double r2 = dx * dx + dy * dy + dz * dz;
			if (r2 < 0.7 * lj_arr->s2_m[i][j])
				goto invalid_step;
			double inv_r2 = 1.0 / r2;
			double inv_r6 = inv_r2 * inv_r2 * inv_r2;
			double inv_r12 = inv_r6 * inv_r6;

			double c6 = lj_arr->c6_m[i][j];
			double c12 = lj_arr->c12_m[i][j];
			
			double term12 = c12 * inv_r12;
			double term6 = c6 * inv_r6;

			energy += term12 - term6;

			// gradient calc
			double dV_ij_over_dx_i = 6.0 * inv_r2 * ( - 2.0 * term12 + term6);

			double gx = dV_ij_over_dx_i * dx;
			double gy = dV_ij_over_dx_i * dy;
			double gz = dV_ij_over_dx_i * dz;

			energy_gradient_arr_out[3 * i] += gx;
			energy_gradient_arr_out[3 * i + 1] += gy;
			energy_gradient_arr_out[3 * i + 2] += gz;

			// ∂V/dxᵢ = - ∂V/dxⱼ
			energy_gradient_arr_out[3 * j] -= gx;
			energy_gradient_arr_out[3 * j + 1] -= gy;
			energy_gradient_arr_out[3 * j + 2] -= gz;
		}
	}
	// calculate the squared gradient norm: ||∇E||² = ∑g²
	for (int d = 0; d < 3 * atomn; d++) {
		double g = energy_gradient_arr_out[d];
		grad_norm_sq += g * g;
	}
	*energy_out = energy;
	*grad_norm_sq_out = grad_norm_sq;
	return 0;

	invalid_step:
	return 1;
}

int lj_energy_gradient(double* atom_arr, int atomn, LJ_Const* lj_arr, double* energy_gradient_arr) {

	for (int i = 0; i < atomn; i++) {
		for (int j = i + 1; j < atomn; j++) {
			double dx = (atom_arr[3 * j] - atom_arr[3 * i]);
			double dy = (atom_arr[3 * j + 1] - atom_arr[3 * i + 1]);
			double dz = (atom_arr[3 * j + 2] - atom_arr[3 * i + 2]);
			double r2 = dx * dx + dy * dy + dz * dz;
			if (r2 < 0.5 * lj_arr->s2_m[i][j])
				return 1;
			double inv_r2 = 1.0 / r2;
			double inv_r6 = inv_r2 * inv_r2 * inv_r2;
			double inv_r12 = inv_r6 * inv_r6;
			
			double c6 = lj_arr->c6_m[i][j];
			double c12 = lj_arr->c12_m[i][j];

			double dV_ij_over_dx_i = (- 2.0) * c12 * inv_r12 + c6 * inv_r6;
			
			double gx = dV_ij_over_dx_i * dx;
			double gy = dV_ij_over_dx_i * dy;
			double gz = dV_ij_over_dx_i * dz;

			energy_gradient_arr[3 * i] += gx;
			energy_gradient_arr[3 * i + 1] += gy;
			energy_gradient_arr[3 * i + 2] += gz;

			// ∂V/dxᵢ = - ∂V/dxⱼ
			energy_gradient_arr[3 * j] = - gx;
			energy_gradient_arr[3 * j + 1] += gy;
			energy_gradient_arr[3 * j + 2] += gz;
		}
	}
	return 0;
}

int just_lj_energy(double* atom_arr, int atomn, LJ_Const* lj_arr, double* energy_out) {

	double energy = 0;
	for (int i = 0; i < atomn; i++) {
		for (int j = i + 1; j < atomn; j++) {
			double dx = (atom_arr[3 * i] - atom_arr[3 * j]);
			double dy = (atom_arr[3 * i + 1] - atom_arr[3 * j + 1]);
			double dz = (atom_arr[3 * i + 2] - atom_arr[3 * j + 2]);
			double r2 = dx * dx + dy * dy + dz * dz;
			if (r2 < 0.1 * lj_arr->s2_m[i][j])
				goto invalid_step;
			double inv_r2 = 1.0 / r2;
			double inv_r6 = inv_r2 * inv_r2 * inv_r2;
			double inv_r12 = inv_r6 * inv_r6;

			double c6 = lj_arr->c6_m[i][j];
			double c12 = lj_arr->c12_m[i][j];
			
			double term12 = c12 * inv_r12;
			double term6 = c6 * inv_r6;

			energy += term12 - term6;
		}
	}
	*energy_out = energy;
	return 0;

	invalid_step:
	return 1;
}


double rms_p1_p2(double* a, double* b, int atomn, int uniq_comb) {
	double rms = 0;
	for (int i = 0; i < atomn - 1; i++) {
		for (int j = i + 1; j < atomn; j++) {
			double da_x = a[3 * i] - a[3 * j];
			double da_y = a[3 * i + 1] - a[3 * j + 1];
			double da_z = a[3 * i + 2] - a[3 * j + 2];
			double da = sqrt(da_x * da_x + da_y * da_y + da_z * da_z);

			double db_x = b[3 * i] - b[3 * j];
			double db_y = b[3 * i + 1] - b[3 * j + 1];
			double db_z = b[3 * i + 2] - b[3 * j + 2];
			double db = sqrt(db_x * db_x + db_y * db_y + db_z * db_z);

			double dp = da - db;
			rms += dp * dp;
		}
	}
	rms /= uniq_comb;
	rms = sqrt(rms);

	return rms;
}

double p1_p2_norm_sq(double* p1, double* p2, int dim) {
	double norm_sq = 0;
	for (int d = 0; d < dim; d++) {
		double dd = p1[d] - p2[d];
		norm_sq += dd * dd;
	}
	return norm_sq;
}

double dist_norm_sq(double* p, int atomn) {
	// calculate the lower triangular distance matrix squared norm for a particle
	double d_p_norm_sq = 0;
	for (int i = 0; i < atomn; i++) {
		for (int j = 1; j < atomn; j++) {
			double dx = p[i] - p[j];
			double dy = p[i + 1] - p[j + 1];
			double dz = p[i + 2] - p[j + 2];
			d_p_norm_sq += dx * dx + dy * dy + dz * dz;
		}
	}
	d_p_norm_sq /= atomn;
	return d_p_norm_sq;
}
