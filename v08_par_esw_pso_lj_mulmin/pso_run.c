#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <omp.h>
#include <mkl.h>
#include "main.h"
#include "lj.h"
#include "pso_run.h"
#include "universal.h"

static double* ref_score = NULL;

int sort_elites(const void* a, const void* b) {
	int idx_a = *(const int*) a;
	int idx_b = *(const int*) b;

	if (ref_score[idx_a] < ref_score[idx_b]) return -1;
	if (ref_score[idx_a] > ref_score[idx_b]) return 1;
	return 0;
}

double score_f(double e, double e_min, double e_range, double g, double g_min, double g_range) {
	double e_term = (e - e_min) / e_range;
	double g_term = (g - g_min) / g_range;

	// return (0.1 * e_term + g_term);
	return (e);
}

Swarm* pso_run(Atoms* atoms, int swarm_size, int max_iter) {

	// ---[ one off stuff ]-----------------------------------------
	double max_x_perturbation = 2.5;
	double max_v_perturbation = 0.1;

	int eliten = 6;
	int elites[eliten];
	int is_elite[swarm_size], indices[swarm_size];
	for (int p = 0; p < swarm_size; p++) {
		is_elite[p] = 0;
		indices[p] = p;
	}
	double it_eigvs[3 * swarm_size];

	int atomn = atoms->atomn;
	int dim = 3 * atomn;
	size_t particle_size = dim * sizeof(double);
	int uniq_comb = (atomn * atomn - atomn) / 2;


	// pre calc constants for init guess
	Constants* const_arr = precalc_const(atoms);
	double mean_sigma = 0, mean_epsilon = 0;
	for (int i = 0; i < atomn; i++)
		for (int j = i + 1; j < atomn; j++) {
			mean_sigma += const_arr->s_m[atomn * i + j];
			mean_epsilon += const_arr->e_m[atomn * i + j];
		}
	mean_sigma /= uniq_comb;
	mean_epsilon /= uniq_comb;

	double mol_weight = 0;
	for (int i = 0; i < atomn; i++)
		mol_weight += const_arr->weights_vec[i];

	// center the guess geom
	double particle_com[swarm_size * dim];
	calc_com(atoms->coords, atomn, const_arr->weights_vec, mol_weight, &particle_com[0]);
	shift_to(atoms->coords, &particle_com[0], dim);

	// ===[ INIT SWARMS ]===========================================

	Swarm* swarm = malloc(sizeof(Swarm) + (6 * swarm_size * dim + 6 * swarm_size) * sizeof(double));
	if (!swarm) exit(1);

	double* mem = (double*) (swarm + 1);
	swarm->x = mem; mem += swarm_size * dim;
	swarm->v = mem; mem += swarm_size * dim;
	swarm->g = mem; mem += swarm_size * dim;
	swarm->pbest_x = mem; mem += swarm_size * dim;
	swarm->pbest_g = mem; mem += swarm_size * dim;
	swarm->pbest_e = mem; mem += swarm_size;
	swarm->pbest_f = mem; mem += swarm_size;
	swarm->pbest_g_norm_sq = mem; mem += swarm_size;
	swarm->prev_iter_pbest_x = mem; mem += swarm_size * dim;
	swarm->prev_iter_pbest_e = mem; mem += swarm_size;
	swarm->prev_iter_pbest_f = mem; mem += swarm_size;
	swarm->prev_iter_pbest_g_norm_sq = mem; mem += swarm_size;

	// kabsch is needed for every particle so carve a block beforehand
	size_t struct_mem = swarm_size * sizeof(Kabsch_Arrays);
	size_t arr_mem = swarm_size * ((48 + 3 * atomn) * sizeof(double));
	size_t block_size = struct_mem + arr_mem;
	Kabsch_Arrays* kabsch_block = calloc(1, block_size);
	if (!kabsch_block) exit(1);

	mem = (double*)((char*) kabsch_block + struct_mem);
	for (int p = 0; p < swarm_size; p++) {
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
	unsigned int swarm_seeds[swarm_size];
	for (int p = 0; p < swarm_size; p++)
		swarm_seeds[p] = (unsigned int) rand();

	// ---[ parallel init ]-----------------------------------------
	#pragma omp parallel for schedule(guided, 2)
	for (int p = 0; p < swarm_size; p++) {

		double* x_p = &swarm->x[p * dim];
		double* v_p = &swarm->v[p * dim];
		double* g_p = &swarm->g[p * dim];
		double* pbest_x_p = &swarm->pbest_x[p * dim];
		double* pbest_g_p = &swarm->pbest_g[p * dim];
		double* pbest_e_p = &swarm->pbest_e[p];
		double* pbest_g_norm_sq_p = &swarm->pbest_g_norm_sq[p];
		double* prev_iter_pbest_x_p = &swarm->prev_iter_pbest_x[p * dim];
		double* prev_iter_pbest_e_p = &swarm->prev_iter_pbest_e[p];
		double* prev_iter_pbest_g_norm_sq_p = &swarm->prev_iter_pbest_g_norm_sq[p];

		re_init:
		for (int d = 0; d < dim; d++) {
			x_p[d] = atoms->coords[d] + ((double) rand_r(&swarm_seeds[p]) / RAND_MAX) * 2 * max_x_perturbation - max_x_perturbation;
			v_p[d] = ((double) rand_r(&swarm_seeds[p]) / RAND_MAX) * 2 * max_v_perturbation - max_v_perturbation;
		}

		// center to com, find optimal rot matrix to align the particle with guess geom and then rotate both the particle & its velocity
		calc_com(x_p, atomn, const_arr->weights_vec, mol_weight, &particle_com[dim * p]);
		shift_to(x_p, &particle_com[dim * p], dim);
		shift_to(v_p, &particle_com[dim * p], dim);

		double energy, g_norm_sq_p;
		int invalid_step = lennard_jones(CALC_GRAD, x_p, atomn, const_arr, &energy, g_p, &g_norm_sq_p, NULL);
		if (invalid_step)
			goto re_init;

		// update bests
		memcpy(pbest_x_p, x_p, particle_size);
		memcpy(prev_iter_pbest_x_p, x_p, particle_size);
		memcpy(pbest_g_p, g_p, particle_size);
		*pbest_e_p = energy;
		*prev_iter_pbest_e_p = energy;
		*pbest_g_norm_sq_p = g_norm_sq_p;
		*prev_iter_pbest_g_norm_sq_p = g_norm_sq_p;

	}
	// ---[ serial scoring ]----------------------------------------
	// scores are normalized across the values range so everything needs to be already computed
	double e_min = swarm->pbest_e[0], e_max = swarm->pbest_e[0];
	double g_min = swarm->pbest_g_norm_sq[0], g_max = swarm->pbest_g_norm_sq[0];

	for (int p = 1; p < swarm_size; p++) {
		double e = swarm->pbest_e[p];
		double g = swarm->pbest_g_norm_sq[p];
		e_min = (e < e_min) ? e : e_min;
		e_max = (e > e_max) ? e : e_max;
		g_min = (g < g_min) ? g : g_min;
		g_max = (g > g_max) ? g : g_max;
	}
	double e_range = e_max - e_min;
	e_range = (e_range < 1E-12) ? 1.0 : e_range;
	double g_range = g_max - g_min;
	g_range = (g_range < 1E-12) ? 1.0 : g_range;

	for (int p = 0; p < swarm_size; p++) {
		double* pbest_e_p = &swarm->pbest_e[p];
		double* pbest_g_norm_sq_p = &swarm->pbest_g_norm_sq[p];
		
		double score = score_f(*pbest_e_p, e_min, e_range, *pbest_g_norm_sq_p, g_min, g_range);

		swarm->pbest_f[p] = score;
		swarm->prev_iter_pbest_f[p] = score;
	}

	// ---[ serial elites init ]------------------------------------
	ref_score = swarm->pbest_f;
	qsort(indices, swarm_size, sizeof(int), sort_elites);

	for (int k = 0; k < eliten; k++) {
		is_elite[indices[k]] = 1;
		elites[k] = indices[k];
	}
	// =============================================================


	// =============================================================
	// ===[ PSO OPT RUN ]===========================================
	// =============================================================


	// ---[ pso params ]--------------------------------------------
	double w_start = 0.9, w_end = 0.4;
	double v_max_start = 0.6 * mean_sigma, v_max_end = 0.05 * mean_sigma;
	int normal_step = 0 * max_iter / 4;
	int explor_iter_max = 4000;
	int explor_iter = (max_iter < explor_iter_max) ? max_iter : explor_iter_max;
	// -------------------------------------------------------------

	for (int iter = 0; iter < max_iter; iter++) {

		double pso_progress = (iter < explor_iter) ? (double) iter / explor_iter : 1;
		double end_of_run_factor = pso_progress;

		double w = w_start - (w_start - w_end) * end_of_run_factor;
		double v_max = v_max_start - (v_max_start - v_max_end) * end_of_run_factor;

		double phi_max = 24 * (1 - w * w) / (7 - 5 * w);
		double phi = 0.8 * phi_max;

		double a1_start = 0.5, a1_end = 1 - a1_start;
		double a2_start = 0.5, a2_end = 1 - a2_start;
		double a1 = a1_start - (a1_start - a1_end) * end_of_run_factor;
		double a2 = a2_start - (a2_start - a2_end) * end_of_run_factor;

		double c1 = a1 * phi;
		double c2 = a2 * phi;
		// double cg = 0.5;

		double similarity_thresh_start = 0.04;
		double similarity_thresh_end = 0.004;
		double similarity_thresh = similarity_thresh_start - (similarity_thresh_start - similarity_thresh_end) * end_of_run_factor;

		// int skip_g_comp = 1;
		// if (iter < normal_step)
		// 	skip_g_comp = 1;

		// ---[ parallel run ]------------------------------------------
		#pragma omp parallel for schedule(guided, 2)
		for (int p = 0; p < swarm_size; p++) {

			double* x_p = &swarm->x[p * dim];
			double* v_p = &swarm->v[p * dim];
			double* g_p = &swarm->g[p * dim];
			double* pbest_x_p = &swarm->pbest_x[p * dim];
			double* pbest_g_p = &swarm->pbest_g[p * dim];
			double* pbest_e_p = &swarm->pbest_e[p];
			double* pbest_f_p = &swarm->pbest_f[p];
			double* pbest_g_norm_sq_p = &swarm->pbest_g_norm_sq[p];
			double* prev_iter_pbest_x_p = &swarm->prev_iter_pbest_x[p * dim];
			double* prev_iter_pbest_e_p = &swarm->prev_iter_pbest_e[p];
			double* prev_iter_pbest_f_p = &swarm->prev_iter_pbest_f[p];
			double* prev_iter_pbest_g_norm_sq_p = &swarm->prev_iter_pbest_g_norm_sq[p];

			// ---[ ring-n topology ]-------------------------------------------
			int ringn = 1;
			int lbest_idx = p;
			double lbest_f_p = swarm->prev_iter_pbest_f[p];
			for (int n = - ringn; n < ringn + 1; n++) {
				int idx = (n < 0) ? (p + n + swarm_size) % swarm_size : (p + n) % swarm_size;
				if (idx == p) continue;

				if (!is_elite[idx] && swarm->prev_iter_pbest_f[idx] < lbest_f_p) {
					lbest_f_p = swarm->prev_iter_pbest_f[idx];
					lbest_idx = idx;
				}
			}
			double* lbest_x_p = &swarm->prev_iter_pbest_x[lbest_idx * dim];
			double c2_p = (is_elite[p]) ? 0 : c2;
			// double cg_p = (is_elite[p]) ? 0 : cg;

			// // ---[ d-ring topology ]---------------------------------------
			// int next_particle_idx = (p + 1) % swarm_size;
			// double* lbest_x_p = &swarm->prev_iter_pbest_x[next_particle_idx * dim];
			//
			// double local_c2 = (is_elite[next_particle_idx]) ? 0 : c2;

			// ---[ regular pso w full topology ]---------------------------
			// // seems to work fine and quicker than prev versions of regular pso
			//
			// // find the neigh lbest
			// // lbests are not alloced arrs
			// int lbest_idx = p;
			// double lbest_f_p = swarm->prev_iter_pbest_f[p];
			// for (int n = 0; n < swarm_size; n++) {
			// 	if (swarm->prev_iter_pbest_f[n] < lbest_f_p) {
			// 		lbest_f_p = swarm->prev_iter_pbest_f[n];
			// 		lbest_idx = n;
			// 	}
			// }
			// double* lbest_x_p = &swarm->prev_iter_pbest_x[lbest_idx * dim];
			// double local_c2 = c2;

			// store prev positions to revert if the step results in a collisoin
			double prev_x_p[dim];
			double prev_v_p[dim];
			double prev_g_p[dim];
			memcpy(prev_x_p, x_p, particle_size);
			memcpy(prev_v_p, v_p, particle_size);
			memcpy(prev_g_p, g_p, particle_size);

			// calc velocity update
			double v_pos_norm_sq = 0;
			double v_pos_p[dim];
			unsigned int* seed = &swarm_seeds[p];
			for (int d = 0; d < dim; d++) {
				double r1 = (double) rand_r(seed) / RAND_MAX;
				double r2 = (double) rand_r(seed) / RAND_MAX;

				v_pos_p[d] = w * v_p[d];
				v_pos_p[d] += c1 * r1 * (prev_iter_pbest_x_p[d] - x_p[d]);
				v_pos_p[d] += c2_p * r2 * (lbest_x_p[d] - x_p[d]);

				v_pos_norm_sq += v_pos_p[d] * v_pos_p[d];

				// v_p[d] *= w;
				// v_p[d] += c1 * r1 * (prev_iter_pbest_x_p[d] - x_p[d]);
				// v_p[d] += c2_p * r2 * (lbest_x_p[d] - x_p[d]);
				//
				// v_p[d] = (v_p[d] > v_max) ? v_max : v_p[d];
				// v_p[d] = (v_p[d] < - v_max) ? - v_max : v_p[d];
				//
				// // update positions
				// x_p[d] += v_p[d];
			}

			// ---[ gradient term calc ]------------------------------------
			// double g_comp[dim];
			// if (!skip_g_comp) {
			//
			// 	// find closest elite
			// 	double it_eigvs_p[6];
			// 	int closest_elite_idx = elites[0];
			// 	calc_inertia_tens_eigv(x_p, atomn, const_arr->weights_vec, &it_eigvs_p[0]);
			// 	calc_inertia_tens_eigv(&swarm->prev_iter_pbest_x[elites[0] * dim], atomn, const_arr->weights_vec, &it_eigvs_p[3]);
			// 	double min_sim = it_eigv_similarity_check(&it_eigvs_p[0], &it_eigvs_p[3]);
			// 	for (int e = 1; e < eliten; e++) {
			// 		calc_inertia_tens_eigv(&swarm->prev_iter_pbest_x[elites[e] * dim], atomn, const_arr->weights_vec, &it_eigvs_p[3]);
			// 		double sim = it_eigv_similarity_check(&it_eigvs_p[0], &it_eigvs_p[3]);
			// 		if (sim < min_sim) {
			// 			min_sim = sim;
			// 			closest_elite_idx = elites[e];
			// 		}
			// 	}
			//
			// 	// create a local copy of that elite and align it with p
			// 	double aligned_elite[dim];
			// 	memcpy(aligned_elite, &swarm->prev_iter_pbest_x[closest_elite_idx * dim], dim * sizeof(double));
			// 	kabsch_optimal_rot(x_p, aligned_elite, atomn, kabsch_block);
			// 	kabsch_rotation(aligned_elite, atomn, kabsch_block);
			//
			// 	// calc g_comp = g - g * ((x - x_min) / |x - x_min|²) * (x - x_min)
			// 	// = g - (g * u / u²) * u, parenthesis is a scalar
			// 	double g_comp_norm_sq = 0; 
			// 	double u_d[dim];
			// 	double g_dot_u = 0;
			// 	double u_dot_u = 0;
			// 	for (int d = 0; d < dim; d++) {
			// 		u_d[d] = x_p[d] - aligned_elite[d];
			// 		g_dot_u += g_p[d] * u_d[d];
			// 		u_dot_u += u_d[d] * u_d[d];
			// 	}
			// 	if (u_dot_u < 1E-12)
			// 		for (int d = 0; d < dim; d++)
			// 			g_comp[d] = 0;
			// 	else {
			// 		for (int d = 0; d < dim; d++) {
			// 			g_comp[d] = g_p[d] - (g_dot_u / u_dot_u) * u_d[d];
			// 			g_comp_norm_sq += g_comp[d] * g_comp[d];
			// 		}
			// 	}
			//
			// 	// extra normalization step in case g_comp is bigger than the rest
			// 	if (g_comp_norm_sq > v_pos_norm_sq && g_comp_norm_sq > 1E-12) {
			// 		double scale = sqrt(v_pos_norm_sq / g_comp_norm_sq);
			// 		for (int d = 0; d < dim; d++)
			// 			g_comp[d] *= scale;
			// 	}
			// }
			// -------------------------------------------------------------

			// update velocity
			for (int d = 0; d < dim; d++) {
				// if (skip_g_comp)
				// 	v_p[d] = v_pos_p[d];
				// else {
				// 	double rg = (double) rand_r(seed) / RAND_MAX;
				// 	v_p[d] = v_pos_p[d] + cg_p * rg * g_comp[d];
				// }

				v_p[d] = (v_p[d] > v_max) ? v_max : v_p[d];
				v_p[d] = (v_p[d] < - v_max) ? - v_max : v_p[d];

				// update positions
				x_p[d] += v_p[d];
			}

			// center to com
			calc_com(x_p, atomn, const_arr->weights_vec, mol_weight, &particle_com[dim * p]);
			shift_to(x_p, &particle_com[dim * p], dim);
			shift_to(v_p, &particle_com[dim * p], dim);

			// calc lj energy for particle p and check for step validity
			// if the step is invalid send the velocity to the oposite direction and half it
			// while keeping previous pos and energy
			double energy, g_norm_sq_p;
			int invalid_step = lennard_jones(CALC_GRAD, x_p, atomn, const_arr, &energy, g_p, &g_norm_sq_p, NULL);
			if (invalid_step) {
				for (int d = 0; d < dim; d++) {
					v_p[d] = - 0.5 * prev_v_p[d];
					x_p[d] = prev_x_p[d];
					g_p[d] = prev_g_p[d];
				}
				continue;
			}

			double score = score_f(energy, e_min, e_range, g_norm_sq_p, g_min, g_range);
			double prev_score = score_f(*prev_iter_pbest_e_p, e_min, e_range, *prev_iter_pbest_g_norm_sq_p, g_min, g_range);

			if (score < prev_score) {
				*pbest_e_p = energy;
				*pbest_g_norm_sq_p = g_norm_sq_p;
				*pbest_f_p = score;
				memcpy(pbest_x_p, x_p, particle_size);
				calc_inertia_tens_eigv(x_p, atomn, const_arr->weights_vec, &it_eigvs[3 * p]);
			}
		}
		// ---[ any serial job that needs to be done ]------------------

		// update pbest of prev iter
		memcpy(swarm->prev_iter_pbest_x, swarm->pbest_x, swarm_size * dim * sizeof(double));
		memcpy(swarm->prev_iter_pbest_e, swarm->pbest_e, swarm_size * sizeof(double));
		memcpy(swarm->prev_iter_pbest_g_norm_sq, swarm->pbest_g_norm_sq, swarm_size * sizeof(double));
		memcpy(swarm->prev_iter_pbest_f, swarm->pbest_f, swarm_size * sizeof(double));
		for (int p = 0; p < swarm_size; p++)
			is_elite[p] = 0;

		// update minmaxes & ranges
		e_min = swarm->pbest_e[0], e_max = swarm->pbest_e[0];
		g_min = swarm->pbest_g_norm_sq[0], g_max = swarm->pbest_g_norm_sq[0];

		for (int p = 1; p < swarm_size; p++) {
			double e = swarm->pbest_e[p];
			double g = swarm->pbest_g_norm_sq[p];
			e_min = (e < e_min) ? e : e_min;
			e_max = (e > e_max) ? e : e_max;
			g_min = (g < g_min) ? g : g_min;
			g_max = (g > g_max) ? g : g_max;
		}
		e_range = e_max - e_min;
		e_range = (e_range < 1E-12) ? 1.0 : e_range;
		g_range = g_max - g_min;
		g_range = (g_range < 1E-12) ? 1.0 : g_range;

		for (int p = 0; p < swarm_size; p++) {
			double* pbest_e_p = &swarm->pbest_e[p];
			double* pbest_g_norm_sq_p = &swarm->pbest_g_norm_sq[p];

			double score = score_f(*pbest_e_p, e_min, e_range, *pbest_g_norm_sq_p, g_min, g_range);

			swarm->pbest_f[p] = score;
			swarm->prev_iter_pbest_f[p] = score;
		}

		// update elites arr
		ref_score = swarm->pbest_f;
		qsort(indices, swarm_size, sizeof(int), sort_elites);

		// set the condition for the elites to be a different structure than the already existing ones
		elites[0] = indices[0];
		is_elite[indices[0]] = 1;
		int elite_count = 1, p = 1;
		while (elite_count < eliten && p < swarm_size) {
			int unique_elite = 1;
			for (int k = 0; k < elite_count; k++)
				if (it_eigv_similarity_check(&it_eigvs[3 * indices[p]], &it_eigvs[3 * elites[k]]) < similarity_thresh) {
					unique_elite = 0;
					break;
				}
			if (unique_elite) {
				elites[elite_count] = indices[p]; 
				is_elite[indices[p]] = 1;
				elite_count++;
			}
			p++;
		}
	}

	// fclose(wstream);

	// =============================================================
	// ===[ PSO OPT RUN END ]=======================================
	// =============================================================

	double uniqueness_m[swarm_size * swarm_size];
	for (int i = 0; i < swarm_size; i++) {
		double* coefs_p1 = &it_eigvs[3 * i];
		for (int j = 0; j < swarm_size; j++) {
			double* coefs_p2 = &it_eigvs[3 * j];
			double eigv_norm_sq = it_eigv_similarity_check(coefs_p1, coefs_p2);
			uniqueness_m[swarm_size * i + j] = eigv_norm_sq;
		}
	}
	matrix_print_flat("uniqueness matrix:\n", uniqueness_m, swarm_size, swarm_size);

	printf("\nswarm converged to:\n");
	for (int k = 0; k < swarm_size; k++) {

		double energy;
		double G[dim];
		double g_norm_sq;
		double H[dim * dim];
		double eigv[dim];
		lennard_jones(CALC_HESS, &swarm->pbest_x[indices[k] * dim], atomn, const_arr, &energy, G, &g_norm_sq, H);

		LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', dim, H, dim, eigv);

		printf("rank %3d (p = %3d): E = %12.8f kj/mol, ||∇E||² = %12.8f, 1st eigv(H) = %8.4f, 6th eigv(H) =  %8.4f",
		k + 1, indices[k] + 1, swarm->pbest_e[indices[k]], swarm->pbest_g_norm_sq[indices[k]], eigv[0], eigv[5]);
		if (is_elite[indices[k]])
			printf("\t <--- is elite\n");
		else printf("\n");
	}

	printf("\nbest geom:\n");
	for (int i = 0; i < atomn; i++) {
		printf("%s\t%12.8f\t%12.8f\t%12.8f\n", atoms->types[i], swarm->pbest_x[indices[0] * i], swarm->pbest_x[indices[0] * i + 1], swarm->pbest_x[indices[0] * i + 2]);
	}
	printf("\n");

	FILE* wfinal = fopen("final_geoms.xyz", "w");
	fprintf(wfinal, "%d\n\n", atomn * swarm_size);
	for (int p = 0; p < swarm_size; p++) {
		double* pbest_x_p = &swarm->pbest_x[p * dim];
		for (int a = 0; a < atomn; a++) {
			fprintf(wfinal, "P%d\t%12.8f\t%12.8f\t%12.8f\n", p + 1, pbest_x_p[3 * a], pbest_x_p[3 * a + 1], pbest_x_p[3 * a + 2]);
		}
	}
	fclose(wfinal);

	free(kabsch_block);
	free(const_arr);

	return swarm;
}
