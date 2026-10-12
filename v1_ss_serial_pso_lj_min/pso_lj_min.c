#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mkl.h>
#include "main.h"
#include "lj.h"
#include "pso_lj_min.h"
#include "mkl-math.h"

void shift_to_centroid(double* vector_arr, int atomn) {
	double x = 0, y = 0, z = 0;
	for (int i = 0; i < atomn; i++) {
		x += vector_arr[3 * i];
		y += vector_arr[3 * i + 1];
		z += vector_arr[3 * i + 2];
	}
	x /= atomn, y /= atomn, z /= atomn;
	for (int i = 0; i < atomn; i++) {
		vector_arr[3 * i] -= x;
		vector_arr[3 * i + 1] -= y;
		vector_arr[3 * i + 2] -= z;
	}
}

typedef struct {
	double* h; // 3x3
	double* s; // 3
	double* u; // 3x3
	double* vt; // 3x3
	double* r; // 3x3
	double* rotated; // atomn x 3
	double* i; // 3x3 id for check
} Kabsch_Arrays;

Kabsch_Arrays* malloc_for_kabsch(int atomn) {
	size_t block_size = sizeof(Kabsch_Arrays) + (9 + 3 + 9 + 9 + 9 + 3 * atomn + 9) * sizeof(double);
	Kabsch_Arrays* kabsch_block = calloc( 1, block_size);
	// H = 3 x 3, S = 3, U = 3 x 3, V^T = 3 x 3, R = 3 x 3, rotated = atomn x 3
	double* mem = (double*)(kabsch_block + 1);

	kabsch_block->h = mem; mem += 9;
	kabsch_block->s = mem; mem += 3;
	kabsch_block->u = mem; mem += 9;
	kabsch_block->vt = mem; mem += 9;
	kabsch_block->r = mem; mem += 9;
	kabsch_block->rotated = mem; mem += 3 * atomn;
	kabsch_block->i = mem; mem += 9;

	return kabsch_block;
}

void kabsch_alignment(double* ref_arr, double* arr_to_align, int atomn, Kabsch_Arrays* kabsch_block, double* velocity) {

	double* q = ref_arr;
	double* p = arr_to_align;
	double* v = velocity;
	double* h = kabsch_block->h;
	double* s = kabsch_block->s;
	double* u = kabsch_block->u;
	double* vt = kabsch_block->vt;
	double* r = kabsch_block->r;
	double* rotated = kabsch_block->rotated;
	double* id = kabsch_block->i;

	id[0] = 1.0; id[4] = 1.0; id[8] = 1.0;

	// calc covariance matrix H = P^T * Q (3x3)
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, atomn, 1.0, p, 3, q, 3, 0.0, h, 3);
	// perform SVD on H, H = USV^T
	// svd is in fortran & column major so U -> V^T and V^T -> U and i need to swap the pointers after svd
	matrix_svd_flat(h, 3, 3, s, u, vt);
	double* dummy = u; u = vt; vt = dummy;
	// calculate optimal rotation matrix R = VU^T
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, 3, 3, 3, 1.0, vt, 3, u, 3, 0.0, r, 3);
	// calc det(R), if < 0 then its an improper rotation and V needs to be inverted (z = -z)
	// and then R recalculated
	double det_r = 	  r[0] * (r[4] * r[8] - r[5] * r[7])
					- r[1] * (r[3] * r[8] - r[5] * r[6])
					+ r[2] * (r[3] * r[7] - r[4] * r[6]);
	if (det_r < 0.0) {
		vt[6] *= - 1; vt[7] *= - 1; vt[8] *= - 1;
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, 3, 3, 3, 1.0, vt, 3, u, 3, 0.0, r, 3);
	}
	// rotate P, P = PR^T
	// (to rotate a row vector its PR^T, PR is for column vectors)
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, atomn, 3, 3, 1.0, p, 3, r, 3, 0.0, rotated, 3);
	for (int i = 0; i < 3 * atomn; i++)
		arr_to_align[i] = rotated[i];
	// also rotate velocity
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, atomn, 3, 3, 1.0, v, 3, r, 3, 0.0, rotated, 3);
	for (int i = 0; i < 3 * atomn; i++)
		v[i] = rotated[i];

	// check R for corruption
	// cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 3, 1.0, r, 3, r, 3, -1.0, id, 3);
	// printf("\n");
	// for (int i = 0; i < 3; i++) {
	// 	for (int j = 0 ; j < 3; j++)
	// 		printf("%18.12f\t", id[3 * i + j]);
	// 	printf("\n");
	// }
	// printf("\n");
}

#define PERTURBATION(i) ((((double) rand() / RAND_MAX) * 2 * i) - i)

Swarm* pso_min_run(Atoms* atoms, int swarm_size, int max_iter) {

	
	double max_x_perturbation = 2.5;
	double max_v_perturbation = 0.1;

	int atomn = atoms->atomn;
	int dim = 3 * atomn;

	// pre calc lj constants for init guess
	LJ_Const* lj_arr = malloc(sizeof(LJ_Const));
	precalc_lj_const(atoms, lj_arr);


	// init swarm
	Swarm* swarm = malloc(sizeof(Swarm) + (3 * swarm_size * dim + swarm_size + dim) * sizeof(double));
	if (!swarm) exit(1);

	double* mem = (double*) (swarm + 1);

	swarm->x = mem; mem += swarm_size * dim;
	swarm->v = mem; mem += swarm_size * dim;
	swarm->pbest_x = mem; mem += swarm_size * dim;
	swarm->pbest_value = mem; mem += swarm_size;
	swarm->gbest_x = mem; mem += dim;

	swarm->gbest_value = DBL_MAX;

	// alloc mem for kabsch alignment
	// each particle is rotated to align best with gbest every cycle
	Kabsch_Arrays* kabsch_block = malloc_for_kabsch(atomn);


	for (int p = 0; p < swarm_size; p++) {
		double* x_p = &swarm->x[p * dim];
		double* v_p = &swarm->v[p * dim];
		double* pbest_x_p = &swarm->pbest_x[p * dim];

		re_init:
		for (int d = 0; d < dim; d++) {
			x_p[d] = atoms->coords[d] + PERTURBATION(max_x_perturbation);
			v_p[d] = PERTURBATION(max_v_perturbation);
		}

		shift_to_centroid(x_p, atomn);
		shift_to_centroid(v_p, atomn);
		kabsch_alignment(atoms->coords, x_p, atomn, kabsch_block, v_p);

		double energy;
		int invalid_step = lj_energy(pbest_x_p, atomn, lj_arr, &energy);
		if (invalid_step) goto re_init;

		memcpy(pbest_x_p, x_p, dim * sizeof(double));

		if (swarm->pbest_value[p] < swarm->gbest_value) {
			swarm->gbest_value = swarm->pbest_value[p];
			memcpy(swarm->gbest_x, pbest_x_p, dim * sizeof(double));
		}
	}

	// ================================== PSO OPT RUN ===================================

	double mean_sigma = 0.0;
	for (int i = 0; i < atomn; i++)
		for (int j = i + 1; j < atomn; j++)
			mean_sigma += lj_arr->s_m[i][j];
	mean_sigma /= atomn;

	// -------------- pso params ------------------
	double c1_start = 2.0, c1_end = 0.8;
	double c2_start = 0.8, c2_end = 2.0;
	double w_start = 0.9, w_end = 0.4;
	double v_max_start = 0.6 * mean_sigma, v_max_end = 0.001 * mean_sigma;
	// --------------------------------------------

	int gbest_update_count = 0;
	int last_iter_that_updated_gbest = 0;

	for (int iter = 0; iter < max_iter; iter++) {

		// w, c1, c2, v_max decay/increase quadratically
		double pso_progress = (double) iter / max_iter;
		double end_of_run_factor = pso_progress * pso_progress;
		double w = w_start - (w_start - w_end) * end_of_run_factor;
		double v_max = v_max_start - (v_max_start - v_max_end) * end_of_run_factor;
		double c1 = c1_start - (c1_start - c1_end) * end_of_run_factor;
		double c2 = c2_start - (c2_start - c2_end) * end_of_run_factor;

		for (int p = 0; p < swarm_size; p++) {
			double* x_p = &swarm->x[p * dim];
			double* v_p = &swarm->v[p * dim];
			double* pbest_x_p = &swarm->pbest_x[p * dim];
			double* gbest_x_p = swarm->gbest_x;

			// store prev positions to revert if the step results in a collisoin
			double prev_x_p[dim];
			double prev_v_p[dim];
			memcpy(prev_x_p, x_p, dim * sizeof(double));
			memcpy(prev_v_p, v_p, dim * sizeof(double));

			// update velocity
			for (int d = 0; d < dim; d++) {
				double r1 = (double) rand() / RAND_MAX;
				double r2 = (double) rand() / RAND_MAX;

				v_p[d] = w * v_p[d] + c1 * r1 * (pbest_x_p[d] - x_p[d]) + c2 * r2 * (gbest_x_p[d] - x_p[d]);

				v_p[d] = (v_p[d] > v_max) ? v_max : v_p[d];
				v_p[d] = (v_p[d] < -v_max) ? -v_max : v_p[d];

				// update positions
				x_p[d] += v_p[d];
			}

			shift_to_centroid(x_p, atomn);
			shift_to_centroid(v_p, atomn);
			kabsch_alignment(swarm->gbest_x, x_p, atomn, kabsch_block, v_p);

			// calc lj energy for particle p
			double energy;
			int invalid_step = lj_energy(x_p, atomn, lj_arr, &energy);
			if (invalid_step) {
				for (int d = 0; d < dim; d++) {
					v_p[d] = - 0.5 * prev_v_p[d];
					x_p[d] = prev_x_p[d];
				}
				continue;
			}

			// update personal & global best
			if (energy < swarm->pbest_value[p]) {
				swarm->pbest_value[p] = energy;
				memcpy(pbest_x_p, x_p, dim * sizeof(double));
				if (energy < swarm->gbest_value) {
					swarm->gbest_value = energy;
					memcpy(gbest_x_p, x_p, dim * sizeof(double));
					gbest_update_count++;
					last_iter_that_updated_gbest = iter;
				}
			}
		}
	}
	// ==================================================================================

	double dist_avg = 0.0;
	for (int p = 0; p < swarm_size; p++) {
		double* x_p = &swarm->x[p * dim];
		double* gbest_x_p = swarm->gbest_x;
		for (int i = 0; i < atomn; i++) {
			double dx = x_p[3 * i] - gbest_x_p[3 * i];
			double dy = x_p[3 * i + 1] - gbest_x_p[3 * i + 1];
			double dz = x_p[3 * i + 2] - gbest_x_p[3 * i + 2];
			dist_avg += sqrt(dx * dx + dy * dy + dz * dz);
		}
	}
	dist_avg /= swarm_size;

	printf("\nbest geom:\n");
	for (int i = 0; i < atomn; i++) {
		printf("%s\t%f\t%f\t%f\n", atoms->types[i], swarm->gbest_x[3 * i], swarm->gbest_x[3 * i + 1], swarm->gbest_x[3 * i + 2]);
	}
	printf("\nLJ potential = %f\n", swarm->gbest_value);
	printf("bgest updated %d times\n", gbest_update_count);
	printf("last iter that gbest updated: %d\n", last_iter_that_updated_gbest);
	printf("swarm distance avg from gbest = %f\n", dist_avg);

	// gradient calc
	double* energy_gradient = calloc(dim, sizeof(double));
	int successful_gradient_calc = lj_energy_gradient(swarm->gbest_x, atomn, lj_arr, energy_gradient);
	if (successful_gradient_calc != 0)
		printf("some atoms were too close and gradient calc did not complete to avoid div by 0\n");
	else {
		printf("\nenergy gradient:\n");
		for (int i = 0; i < atomn; i++) {
			printf("%12.8f\t%12.8f\t%12.8f\n",energy_gradient[3 * i], energy_gradient[3 * i + 1], energy_gradient[3 * i + 2]);
		}
		printf("\n");
	}
	free(energy_gradient);

	free(kabsch_block);
	free(lj_arr->s_m);
	free(lj_arr->s2_m);
	free(lj_arr->c6_m);
	free(lj_arr->c12_m);
	free(lj_arr);

	return swarm;
}
