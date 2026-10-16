#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <mkl.h>
#include <omp.h>
#include "main.h"
#include "lj.h"
#include "pso_run.h"
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

	// 1) calc covariance matrix H = PᵀQ (3x3)
	// 2) perform SVD on H, H = USVᵀ
	// svd is in fortran & column major so U -> Vᵀ and Vᵀ -> U and i need to swap the pointers after svd
	// 3) calculate optimal rotation matrix R = VUᵀ
	// 4) calc det(R), if < 0 then its an improper rotation and V needs to be inverted (z = -z)
	// and then R recalculated
	// 5) rotate P, P = PRᵀ
	// (to rotate a row vector its PRᵀ, PR is for column vectors)
	// 6) also rotate velocity

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, atomn, 1.0, p, 3, q, 3, 0.0, h, 3);
	matrix_svd_flat(h, 3, 3, s, u, vt);
	double* dummy = u; u = vt; vt = dummy;
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, 3, 3, 3, 1.0, vt, 3, u, 3, 0.0, r, 3);
	double det_r = 	  r[0] * (r[4] * r[8] - r[5] * r[7])
					- r[1] * (r[3] * r[8] - r[5] * r[6])
					+ r[2] * (r[3] * r[7] - r[4] * r[6]);
	if (det_r < 0.0) {
		vt[6] *= - 1; vt[7] *= - 1; vt[8] *= - 1;
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, 3, 3, 3, 1.0, vt, 3, u, 3, 0.0, r, 3);
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, atomn, 3, 3, 1.0, p, 3, r, 3, 0.0, rotated, 3);
	for (int i = 0; i < 3 * atomn; i++)
		arr_to_align[i] = rotated[i];
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, atomn, 3, 3, 1.0, v, 3, r, 3, 0.0, rotated, 3);
	for (int i = 0; i < 3 * atomn; i++)
		v[i] = rotated[i];

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


Swarm* pso_run(Atoms* atoms, int sub_swarm_n, int sub_swarm_size, int max_iter) {

	// ---[ one off stuff ]-----------------------------------------
	double max_x_perturbation = 2.5;
	double max_v_perturbation = 0.1;

	int atomn = atoms->atomn;
	int dim = 3 * atomn;
	int particlen = sub_swarm_n * sub_swarm_size;
	size_t particle_size = dim * sizeof(double);
	int uniq_comb = (atomn * atomn - atomn) / 2;

	// center the guess geom
	shift_to_centroid(atoms->coords, atomn);

	// pre calc lj constants for init guess
	LJ_Const* lj_arr = malloc(sizeof(LJ_Const));
	precalc_lj_const(atoms, lj_arr);
	double mean_sigma = 0.0;
	for (int i = 0; i < atomn; i++)
		for (int j = i + 1; j < atomn; j++)
			mean_sigma += lj_arr->s_m[i][j];
	mean_sigma /= atomn;

	// create a bounding box
	double x_max = 0;
	for (int d = 0; d < dim; d++) {
		double x = fabs(atoms->coords[d]);
		x_max = (x > x_max) ? x : x_max;
	}
	x_max = 3 * (x_max + 4);
	// calc the diagonal of the dim dimensional search cube
	double diag = x_max * sqrt(dim);

	// ---[ set up trajectory write ]-----------------------------
	FILE* wstream = fopen("trajectory.xyz", "w");
	char wstream_buffer[64000]; 
	setvbuf(wstream, wstream_buffer, _IOFBF, sizeof(wstream_buffer));
	int log_interval = 10;

	// ===[ INIT SWARMS ]===========================================

	// subswarm: 	x -> sub_swarm_size * dim
	//				v -> sub_swarm_size * dim
	//				g -> sub_swarm_size * dim
	//				pbest_x -> sub_swarm_size * dim
	//				pbest_g -> sub_swarm_size * dim
	//				pbest_e -> sub_swarm_size
	//				g_norm_sq -> sub_swarm_size

	// swarm:		lbest_x -> sub_swarm_n * dim
	//				lbest_g -> sub_swarm_n * dim
	//				lbest_e -> sub_swarm_n
	//				gbest_x -> dim
	//				gbest_g -> dim
	//				gbest_e -> 1 (doesnt need separate alloc)

	SubSwarms* sub_swarm = malloc(sizeof(SubSwarms) + sub_swarm_n * (5 * sub_swarm_size * dim + 2 * sub_swarm_size) * sizeof(double));
	Swarm* swarm = malloc(sizeof(Swarm) + (2 * sub_swarm_n * dim + sub_swarm_n + 2 * dim) * sizeof(double));
	if (!swarm || !sub_swarm) exit(1);

	double* mem = (double*) (sub_swarm + 1);
	sub_swarm->x = mem; mem += sub_swarm_n * sub_swarm_size * dim;
	sub_swarm->v = mem; mem += sub_swarm_n * sub_swarm_size * dim;
	sub_swarm->g = mem; mem += sub_swarm_n * sub_swarm_size * dim;
	sub_swarm->pbest_x = mem; mem += sub_swarm_n * sub_swarm_size * dim;
	sub_swarm->pbest_g = mem; mem += sub_swarm_n * sub_swarm_size * dim;
	sub_swarm->pbest_e = mem; mem += sub_swarm_n * sub_swarm_size;
	sub_swarm->g_norm_sq = mem; mem += sub_swarm_n * sub_swarm_size;

	mem = (double*) (swarm + 1);
	swarm->lbest_x = mem; mem += sub_swarm_n * dim;
	swarm->lbest_g = mem; mem += sub_swarm_n * dim;
	swarm->lbest_e = mem; mem += sub_swarm_n;
	swarm->gbest_x = mem; mem += dim;
	swarm->gbest_g = mem; mem += dim;

	swarm->gbest_e = DBL_MAX;
	for (int s = 0; s < sub_swarm_n; s++)
		swarm->lbest_e[s] = DBL_MAX;
	swarm->gbest_g_norm_sq = DBL_MAX;

	// kabsch is needed for every particle so carve a block beforehand
	size_t struct_mem = particlen * sizeof(Kabsch_Arrays);
	size_t arr_mem = particlen * ((48 + 3 * atomn) * sizeof(double));
	size_t block_size = struct_mem + arr_mem;
	Kabsch_Arrays* kabsch_block = calloc(1, block_size);
	if (!kabsch_block) exit(1);

	mem = (double*)((char*) kabsch_block + struct_mem);
	for (int p = 0; p < particlen; p++) {
		Kabsch_Arrays* kabsch_arr = &kabsch_block[p];
		kabsch_arr->h = mem; mem += 9;
		kabsch_arr->s = mem; mem += 3;
		kabsch_arr->u = mem; mem += 9;
		kabsch_arr->vt = mem; mem += 9;
		kabsch_arr->r = mem; mem += 9;
		kabsch_arr->rotated = mem; mem += 3 * atomn;
		kabsch_arr->i = mem; mem += 9;
	}

	// create a seed for each particle bc rand is not thread safe
	unsigned int swarm_seeds[sub_swarm_n * sub_swarm_size];
	for (int s = 0; s < sub_swarm_n; s++)
		for (int p = 0; p < sub_swarm_size; p++)
			swarm_seeds[s * sub_swarm_size + p] = (unsigned int) rand();

	double* gbest_x = swarm->gbest_x;
	double* gbest_g = swarm->gbest_g;
	double* gbest_e = &swarm->gbest_e;
	double* gbest_g_norm_sq = &swarm->gbest_g_norm_sq;

	// ---[ parallel init ]-----------------------------------------

	// set up locks so that only one particle can update lbest & gbest at a time
	omp_lock_t gbest_lock;
	omp_init_lock(&gbest_lock);
	omp_lock_t lbest_locks[sub_swarm_n];
	for (int s = 0; s < sub_swarm_n; s++)
		omp_init_lock(&lbest_locks[s]);

	#pragma omp parallel for collapse(2) schedule(guided, 2)
	for (int s = 0; s < sub_swarm_n; s++) {
		for (int p = 0; p < sub_swarm_size; p++) {

			double* lbest_x_s = &swarm->lbest_x[s * dim];
			double* lbest_g_s = &swarm->lbest_g[s * dim];
			double* lbest_e_s = &swarm->lbest_e[s];

			int idx_ss = s * sub_swarm_size + p;
			double* x_sp = &sub_swarm->x[idx_ss * dim];
			double* v_sp = &sub_swarm->v[idx_ss * dim];
			double* g_sp = &sub_swarm->g[idx_ss * dim];
			double* pbest_x_sp = &sub_swarm->pbest_x[idx_ss * dim];
			double* pbest_g_sp = &sub_swarm->pbest_g[idx_ss * dim];
			double* pbest_e_sp = &sub_swarm->pbest_e[idx_ss];
			double* g_norm_sq_sp = &sub_swarm->g_norm_sq[idx_ss];

			re_init:
			for (int d = 0; d < dim; d++) {
				x_sp[d] = atoms->coords[d] + ((double) rand_r(&swarm_seeds[idx_ss]) / RAND_MAX) * 2 * max_x_perturbation - max_x_perturbation;
				v_sp[d] = ((double) rand_r(&swarm_seeds[idx_ss]) / RAND_MAX) * 2 * max_v_perturbation - max_v_perturbation;
			}

			shift_to_centroid(x_sp, atomn);
			shift_to_centroid(v_sp, atomn);
			kabsch_alignment(atoms->coords, x_sp, atomn, &kabsch_block[idx_ss], v_sp);

			double energy;
			int invalid_step = lj_energy(x_sp, atomn, lj_arr, g_sp, &energy, g_norm_sq_sp);
			if (invalid_step)
				goto re_init;

			memcpy(pbest_x_sp, x_sp, particle_size);
			memcpy(pbest_g_sp, g_sp, particle_size);
			*pbest_e_sp = energy;

			if (*pbest_e_sp < *lbest_e_s) {
				omp_set_lock(&lbest_locks[s]);
				if (*pbest_e_sp < *lbest_e_s) {
					*lbest_e_s = *pbest_e_sp;
					memcpy(lbest_x_s, pbest_x_sp, particle_size);
					memcpy(lbest_g_s, pbest_g_sp, particle_size);
					if (*pbest_e_sp < *gbest_e) {
						omp_set_lock(&gbest_lock);
						if (*pbest_e_sp < *gbest_e) {
							*gbest_e = *pbest_e_sp;
							memcpy(gbest_x, pbest_x_sp, particle_size);
							memcpy(gbest_g, pbest_g_sp, particle_size);
							*gbest_g_norm_sq = *g_norm_sq_sp;
						}
						omp_unset_lock(&gbest_lock);
					}
				}
				omp_unset_lock(&lbest_locks[s]);
			}
		}
	}
	// =============================================================


	// =============================================================
	// ===[ PSO OPT RUN ]===========================================
	// =============================================================


	// ---[ pso params ]--------------------------------------------
	double w_start = 0.9, w_end = 0.4;
	double v_max_start = 0.6 * mean_sigma, v_max_end = 0.05 * mean_sigma;
	double e = 1.0E-8;
	// -------------------------------------------------------------

	for (int iter = 0; iter < max_iter; iter++) {

		double pso_progress = (double) iter / max_iter;
		double end_of_run_factor = pso_progress;

		double w = w_start - (w_start - w_end) * end_of_run_factor;
		double v_max = v_max_start - (v_max_start - v_max_end) * end_of_run_factor;

		double phi_max = 24 * (1 - w * w) / (7 - 5 * w);
		double phi = 0.8 * phi_max;

		double a1_start = 0.7, a1_end = 1 - a1_start;
		double a2_start = 0.3, a2_end = 1 - a2_start;
		double a1 = a1_start - (a1_start - a1_end) * end_of_run_factor;
		double a2 = a2_start - (a2_start - a2_end) * end_of_run_factor;

		double c1 = a1 * phi;
		double c2 = a2 * phi;
		double cg = 1E-4 * v_max;

		// prev version had parallel subswarms
		// this seems to be worse and threads stay idle so now the hot loops get flattened and let
		// the machine figure out the thread distribution at runtime.
		// now tho each particle needs it own kabsch block

		// ---[ parallel run ]------------------------------------------

		#pragma omp parallel for collapse(2) schedule(guided, 2)
		for (int s = 0; s < sub_swarm_n; s++) {
			for (int p = 0; p < sub_swarm_size; p++) {
				double* lbest_x_s = &swarm->lbest_x[s * dim];
				double* lbest_g_s = &swarm->lbest_g[s * dim];
				double* lbest_e_s = &swarm->lbest_e[s];

				int idx_ss = s * sub_swarm_size + p;
				double* x_sp = &sub_swarm->x[idx_ss * dim];
				double* v_sp = &sub_swarm->v[idx_ss * dim];
				double* g_sp = &sub_swarm->g[idx_ss * dim];
				double* pbest_x_sp = &sub_swarm->pbest_x[idx_ss * dim];
				double* pbest_g_sp = &sub_swarm->pbest_g[idx_ss * dim];
				double* pbest_e_sp = &sub_swarm->pbest_e[idx_ss];
				double* g_norm_sq_sp = &sub_swarm->g_norm_sq[idx_ss];

				unsigned int* seed = &swarm_seeds[idx_ss];

				// store prev positions to revert if the step results in a collisoin
				double prev_x_sp[dim];
				double prev_v_sp[dim];
				memcpy(prev_x_sp, x_sp, dim * sizeof(double));
				memcpy(prev_v_sp, v_sp, dim * sizeof(double));

				// calcuate the diversity of each particle with the lbests of the other swarms.
				// if the current particle is close enough to an existing lbest_x of a different
				// swarm then push that particle away from lbest(p) and pbest(p).
				// set the diversity threshold as well.
				int dir = 1;
				for (int sc = s + 1; sc < sub_swarm_n; sc++) {
					double* lbest_x_sc = &swarm->lbest_x[sc * dim];
					double diversity = rms_p1_p2(x_sp, lbest_x_sc, atomn, uniq_comb);
					diversity /= diag;
					dir = (diversity < 0.005) ? - dir : dir;
					// printf("diversity between particle %d (swarm %d) and lbest %d = %f\n", p, s, sc, diversity);
				}

				// update velocity
				for (int d = 0; d < dim; d++) {
					double r1 = (double) rand_r(seed) / RAND_MAX;
					double r2 = (double) rand_r(seed) / RAND_MAX;
					double rg = (double) rand_r(seed) / RAND_MAX;

					// calcuate the gradient term: g_comp = - g² / (||g||² + ε), (ε = small)
					double g_comp = - (g_sp[d] * g_sp[d]) / (*g_norm_sq_sp + e);

					v_sp[d] *= w;
					v_sp[d] += dir * c1 * r1 * (pbest_x_sp[d] - x_sp[d]);
					v_sp[d] += dir * c2 * r2 * (lbest_x_s[d] - x_sp[d]);
					v_sp[d] += cg * rg * g_comp;

					v_sp[d] = (v_sp[d] > v_max) ? v_max : v_sp[d];
					v_sp[d] = (v_sp[d] < - v_max) ? - v_max : v_sp[d];

					// update positions
					x_sp[d] += v_sp[d];
					x_sp[d] = (x_sp[d] > x_max) ? x_max : x_sp[d];
					x_sp[d] = (x_sp[d] < - x_max) ? - x_max : x_sp[d];
				}

				shift_to_centroid(x_sp, atomn);
				shift_to_centroid(v_sp, atomn);
				kabsch_alignment(gbest_x, x_sp, atomn, &kabsch_block[idx_ss], v_sp);

				// calc lj energy for particle p and check for step validity
				// if the step is invalid send the velocity to the oposite direction and half it
				// while keeping previous pos and energy
				double energy;
				int invalid_step = lj_energy(x_sp, atomn, lj_arr, g_sp, &energy, g_norm_sq_sp);
				if (invalid_step) {
					for (int d = 0; d < dim; d++) {
						v_sp[d] = - 0.5 * prev_v_sp[d];
						x_sp[d] = prev_x_sp[d];
					}
					continue;
				}

				// update personal, local & global best
				if (energy < *pbest_e_sp) {
					*pbest_e_sp = energy;
					memcpy(pbest_x_sp, x_sp, dim * sizeof(double));
					memcpy(pbest_g_sp, g_sp, dim * sizeof(double));
					if (energy < *lbest_e_s) {
						omp_set_lock(&lbest_locks[s]);
						if (energy < *lbest_e_s) {
							*lbest_e_s = energy;
							memcpy(lbest_x_s, x_sp, dim * sizeof(double));
							memcpy(lbest_g_s, g_sp, dim * sizeof(double));
							if (*pbest_e_sp < *gbest_e) {
								omp_set_lock(&gbest_lock);
								if (energy < *gbest_e) {
									*gbest_e = energy;
									memcpy(gbest_x, pbest_x_sp, particle_size);
									memcpy(gbest_g, pbest_g_sp, particle_size);
									*gbest_g_norm_sq = *g_norm_sq_sp;
								}
								omp_unset_lock(&gbest_lock);
							}
						}
						omp_unset_lock(&lbest_locks[s]);
					}
				}
			}
		}
		// ---[ any serial job that needs to be done ]------------------
		// write trajectory.xyz file for vmd
		if (iter % log_interval == 0) {
			fprintf(wstream, "%d\niter = %d\n", atomn * sub_swarm_n, iter);
			for (int s = 0; s < sub_swarm_n; s++) {
				double* lbest_s = &swarm->lbest_x[s * dim];
				for (int a = 0; a < atomn; a++) {
					fprintf(wstream, "S%d\t%12.8f\t%12.8f\t%12.8f\n", s, lbest_s[3 * a], lbest_s[3 * a + 1], lbest_s[3 * a + 2]);
				}
			}
		}
	}
	omp_unset_lock(&gbest_lock);
	for (int s = 0; s < sub_swarm_n; s++)
		omp_unset_lock(&lbest_locks[s]);

	fclose(wstream);

	// =============================================================
	// ===[ PSO OPT RUN END ]=======================================
	// =============================================================

	printf("\nswarms converged to:\n");
	for (int i = 0; i < sub_swarm_n; i++) {
		printf("M%d = %12.8f kj/mol\n", i, swarm->lbest_e[i]);
	}

	for (int s = 0; s < sub_swarm_n; s++) {
		double* lbest_g_s = &swarm->lbest_g[s * dim];
		printf("\nM%d energy gradient:\n", s);
		for (int i = 0; i < atomn; i++) {
			printf("%12.8f\t%12.8f\t%12.8f\n", lbest_g_s[3 * i], lbest_g_s[3 * i + 1], lbest_g_s[3 * i + 2]);
		}
		printf("\n");
	}
	printf("\n");

	printf("best geom gradient norm = %f\n\n", sqrt(swarm->gbest_g_norm_sq));

	free(sub_swarm);
	free(kabsch_block);
	free(lj_arr->s_m);
	free(lj_arr->s2_m);
	free(lj_arr->c6_m);
	free(lj_arr->c12_m);
	free(lj_arr);

	return swarm;
}
