#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mkl.h>
#include "universal.h"
#include "lj.h"
#include "mkl-math.h"


void elem_param_match(char* elem, double* weight, double* sigma, double* epsilon) {
	if (strcmp(elem, "Ar") == 0) {
		*epsilon = 0.997;
		*sigma = 3.4;
		*weight = 39.95;
	} else {
		fprintf(stderr, "unsupported element, exiting now\n");
		exit(1);
	}
}

Constants* precalc_const(Atoms* atoms) {
	int atomn = atoms->atomn;
	double weight1, weight2;
	double epsilon, epsilon1, epsilon2; // kj/mol
	double sigma, sigma1, sigma2; // angstrom

	size_t const_block_size = sizeof(Constants) + (5 * atomn * atomn + atomn) * sizeof(double);
	Constants* const_block = calloc(1, const_block_size);

	double* mem = (double*) (const_block + 1);
	double* weights_vec = mem; mem += atomn;
	double* e_m = mem; mem += atomn * atomn;
	double* s_m = mem; mem += atomn * atomn;
	double* s2_m = mem; mem += atomn * atomn;
	double* c6_m = mem; mem += atomn * atomn;
	double* c12_m = mem; mem += atomn * atomn;

	for (int i = 0; i < atomn; i++) {
		elem_param_match(atoms->types[i], &weight1, &sigma1, &epsilon1);
		weights_vec[i] = weight1;

		for (int j = i; j < atomn; j++) {
			if (strcmp(atoms->types[i], atoms->types[j]) == 0) {
				sigma = sigma1;
				epsilon = epsilon1;
			} else {
				elem_param_match(atoms->types[j], &weight2, &sigma2, &epsilon2);
				sigma = (sigma1 + sigma2) / 2;
				epsilon = sqrt(epsilon1 * epsilon2);
			}
			double sigma_sq = sigma * sigma;
			double sigma_6 = sigma_sq * sigma_sq * sigma_sq;
			double sigma_12 = sigma_6 * sigma_6;

			double c6 = 4 * epsilon * sigma_6;
			double c12 = 4 * epsilon * sigma_12;

			int ij = i * atomn + j;
			int ji = j * atomn + i;

			e_m[ij] = epsilon;
			e_m[ji] = epsilon;
			s_m[ij] = sigma;
			s_m[ji] = sigma;
			s2_m[ij] = sigma_sq;
			s2_m[ji] = sigma_sq;
			c6_m[ij] = c6;
			c6_m[ji] = c6;
			c12_m[ij] = c12;
			c12_m[ji] = c12;

		}
	}
	const_block->weights_vec = weights_vec;
	const_block->e_m = e_m;
	const_block->s_m = s_m;
	const_block->s2_m = s2_m;
	const_block->c6_m = c6_m;
	const_block->c12_m = c12_m;

	return const_block;
}

static inline int lennard_jones_kernel(
	LJ_Mode mode,
	const double* restrict atom_arr,
	const int atomn,
	const Constants* restrict const_arr,
	double* restrict energy_out,
	double* restrict energy_gradient_arr_out,
	double* restrict grad_norm_sq_out,
	double* restrict hessian_out
) {

	if (mode >= CALC_GRAD)
		memset(energy_gradient_arr_out, 0, 3 * atomn * sizeof(double));
	if (mode == CALC_HESS)
		memset(hessian_out, 0, 9 * atomn * atomn * sizeof(double));

	double energy = 0;
	double grad_norm_sq = 0;

	for (int i = 0; i < atomn; i++) {
		for (int j = i + 1; j < atomn; j++) {
			double dx = (atom_arr[3 * i] - atom_arr[3 * j]);
			double dy = (atom_arr[3 * i + 1] - atom_arr[3 * j + 1]);
			double dz = (atom_arr[3 * i + 2] - atom_arr[3 * j + 2]);
			double r2 = dx * dx + dy * dy + dz * dz;
			if (r2 < 0.7 * const_arr->s2_m[atomn * i + j])
				goto invalid_step;
			double inv_r2 = 1.0 / r2;
			double inv_r6 = inv_r2 * inv_r2 * inv_r2;
			double inv_r12 = inv_r6 * inv_r6;

			double c6 = const_arr->c6_m[atomn * i + j];
			double c12 = const_arr->c12_m[atomn * i + j];
			
			double term12 = c12 * inv_r12;
			double term6 = c6 * inv_r6;

			energy += term12 - term6;

			// gradient calc
			double dV_ij_over_dx_i;
			if (mode >= CALC_GRAD) {

				dV_ij_over_dx_i = 6.0 * inv_r2 * ( - 2.0 * term12 + term6);

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

			// hessian calc
			if (mode == CALC_HESS) {

				double d2V_ij_over_dr2 = 6 * inv_r2 * (26 * term12 - 7 * term6);

				double A_ij = dV_ij_over_dx_i;
				double B_ij = (d2V_ij_over_dr2 - A_ij) * inv_r2;
				double delta_x[3] = {dx, dy, dz};

				for (int a = 0; a < 3; a++) {
					for (int b = 0; b < 3; b++) {
						double H_iajb = - B_ij * delta_x[a] * delta_x[b] - (a == b ? A_ij : 0); 
						// off diagonal
						hessian_out[(3 * i + a) * 3 * atomn + (3 * j + b)] = H_iajb;
						hessian_out[(3 * j + a) * 3 * atomn + (3 * i + b)] = H_iajb;
						// diagonal
						hessian_out[(3 * i + a) * 3 * atomn + (3 * i + b)] -= H_iajb;
						hessian_out[(3 * j + a) * 3 * atomn + (3 * j + b)] -= H_iajb;
					}
				}
			}
		}
	}
	*energy_out = energy;

	if (mode >= CALC_GRAD) {
		// calculate the squared gradient norm: ||∇E||² = ∑g²
		for (int d = 0; d < 3 * atomn; d++) {
			double g = energy_gradient_arr_out[d];
			grad_norm_sq += g * g;
		}
		*grad_norm_sq_out = grad_norm_sq;
	}

	return 0;

	invalid_step:
	return 1;
}

int lennard_jones(
	LJ_Mode mode,
	const double* restrict atom_arr,
	const int atomn,
	const Constants* restrict const_arr,
	double* restrict energy_out,
	double* restrict energy_gradient_arr_out,
	double* restrict grad_norm_sq_out,
	double* restrict hessian_out
) {
	switch (mode) {
		case CALC_ENER:
			return lennard_jones_kernel(CALC_ENER, atom_arr, atomn, const_arr, energy_out, NULL,  NULL, NULL);
		case CALC_GRAD:
			return lennard_jones_kernel(CALC_GRAD, atom_arr, atomn, const_arr, energy_out, energy_gradient_arr_out, grad_norm_sq_out, NULL);
		case CALC_HESS:
			return lennard_jones_kernel(CALC_HESS, atom_arr, atomn, const_arr, energy_out, energy_gradient_arr_out, grad_norm_sq_out, hessian_out);
	}
	return 1;
}

void calc_com(double* vector_arr, int atomn, double* weights_vec, double mol_weight, double* com_out) {
	double x = 0, y = 0, z = 0;
	for (int i = 0; i < atomn; i++) {
		x += vector_arr[3 * i] * weights_vec[i];
		y += vector_arr[3 * i + 1] * weights_vec[i];
		z += vector_arr[3 * i + 2] * weights_vec[i];
	}
	x /= mol_weight, y /= mol_weight, z /= mol_weight;

	for (int i = 0; i < atomn; i++) {
		com_out[3 * i] = x;
		com_out[3 * i + 1] = y;
		com_out[3 * i + 2] = z;
	}
}

void shift_to(double* vector_arr, double* dest, int dim) {
	for (int d = 0; d < dim; d++)
		vector_arr[d] -= dest[d];
}

void kabsch_optimal_rot(double* ref_arr, double* arr_to_align, int atomn, Kabsch_Arrays* kabsch_block) {

	double* q = ref_arr;
	double* p = arr_to_align;
	double* h = kabsch_block->h;
	double* s = kabsch_block->s;
	double* u = kabsch_block->u;
	double* vt = kabsch_block->vt;
	double* r = kabsch_block->r;
	double* id = kabsch_block->i;

	id[0] = 1.0; id[4] = 1.0; id[8] = 1.0;

	// 1) calc covariance matrix H = PᵀQ (3x3)
	// 2) perform SVD on H, H = USVᵀ
	// svd is in fortran & column major so U -> Vᵀ and Vᵀ -> U and i need to swap the pointers after svd
	// 3) calculate optimal rotation matrix R = VUᵀ
	// 4) calc det(R), if < 0 then its an improper rotation and V needs to be inverted (z = -z)
	// and then R recalculated

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, atomn, 1.0, p, 3, q, 3, 0.0, h, 3);
	matrix_svd_flat(h, 3, 3, s, u, vt);
	double* dummy = u; u = vt; vt = dummy;
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, 3, 3, 3, 1.0, vt, 3, u, 3, 0.0, r, 3);
	double det_r = 	  r[0] * (r[4] * r[8] - r[5] * r[7])
					- r[1] * (r[3] * r[8] - r[5] * r[6])
					+ r[2] * (r[3] * r[7] - r[4] * r[6]);
	if (det_r < 0) {
		vt[6] *= - 1; vt[7] *= - 1; vt[8] *= - 1;
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, 3, 3, 3, 1.0, vt, 3, u, 3, 0.0, r, 3);
	}

	// check R for corruption
	// R needs to be orthogonal: RRᵀ - I = 0
	// cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 3, 1.0, r, 3, r, 3, -1.0, id, 3);
	// printf("\n");
	// for (int i = 0; i < 3; i++) {
	// 	for (int j = 0 ; j < 3; j++)
	// 		printf("%18.12f\t", id[3 * i + j]);
	// 	printf("\n");
	// }
	// printf("\n");
}

void kabsch_rotation(double* arr_to_rot, int atomn, Kabsch_Arrays* kabsch_block) {
	double* p = arr_to_rot;
	double* r = kabsch_block->r;
	double* rotated = kabsch_block->rotated;

	// 5) rotate P, P = PRᵀ
	// (to rotate a row vector its PRᵀ, PR is for column vectors)

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, atomn, 3, 3, 1.0, p, 3, r, 3, 0.0, rotated, 3);
	for (int i = 0; i < 3 * atomn; i++)
		arr_to_rot[i] = rotated[i];
}

void calc_inertia_tens_eigv(double* x_p, int atomn, double* weights_vec, double* eigv_out) {
	double Ixx = 0, Ixy = 0, Ixz = 0;
	double Iyx = 0, Iyy = 0, Iyz = 0;
	double Izx = 0, Izy = 0, Izz = 0;

	for (int i = 0; i < atomn; i++) {
		double x = x_p[3 * i];
		double y = x_p[3 * i + 1];
		double z = x_p[3 * i + 2];
		double m = weights_vec[i];

		Ixx += m * (y * y + z * z);
		Iyy += m * (x * x + z * z);
		Izz += m * (x * x + y * y);

		Ixy -= m * x * y;
		Ixz -= m * x * z;
		Iyz -= m * y * z;
	}
	Iyx = Ixy;
	Izx = Ixz;
	Izy = Iyz;

	double I[9];
	I[0] = Ixx, I[1] = Ixy, I[2] = Ixz;
	I[3] = Iyx, I[4] = Iyy, I[5] = Iyz;
	I[6] = Izx, I[7] = Izy, I[8] = Izz;

	// calc the eigenvalues
	double x1 = Ixx * Ixx + Iyy * Iyy + Izz * Izz - Ixx * Iyy - Ixx * Izz - Iyy * Izz
				+ 3 * (Iyx * Iyx + Izx * Izx + Izy * Izy);
	double x2 = - (2 * Ixx - Iyy - Izz) * (2 * Iyy - Ixx - Izz) * (2 * Izz - Ixx - Iyy)
				+ 9 * ((2 * Izz - Ixx - Iyy) * Iyx * Iyx + (2 * Iyy - Ixx - Izz) * Izx * Izx
				+ (2 * Ixx - Iyy - Izz) * Izy * Izy) - 54 * Ixy * Iyz * Izx;

	double root = 4 * x1 * x1 * x1 - x2 * x2;
	if (root < 0) root = 0;
	double phi = atan2(sqrt(root), x2);

	double lambda1 = (Ixx + Iyy + Izz - 2 * sqrt(x1) * cos(phi / 3)) / 3;
	double lambda2 = (Ixx + Iyy + Izz + 2 * sqrt(x1) * cos((phi - M_PI) / 3)) / 3;
	double lambda3 = (Ixx + Iyy + Izz + 2 * sqrt(x1) * cos((phi + M_PI) / 3)) / 3;

	// sort them
	if (lambda1 > lambda2) { double t = lambda1; lambda1 = lambda2; lambda2 = t; }
	if (lambda2 > lambda3) { double t = lambda2; lambda2 = lambda3; lambda3 = t; }
	if (lambda1 > lambda2) { double t = lambda1; lambda1 = lambda2; lambda2 = t; }

	eigv_out[0] = lambda1;
	eigv_out[1] = lambda2;
	eigv_out[2] = lambda3;

	// LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', 3, I, 3, eigv_out);
}

double it_eigv_similarity_check(double* coefs_p1, double* coefs_p2) {
	double rel_norm_sq = 0;
	for (int i = 0; i < 3; i++) {
		double rel_diff = (coefs_p1[i] - coefs_p2[i]) / (0.5 * (coefs_p1[i] + coefs_p2[i]) + 1E-8);
		rel_norm_sq += rel_diff * rel_diff;
	}
	rel_norm_sq /= 3;

	return rel_norm_sq;
}
