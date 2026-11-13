#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	double* x;
	double* v;
	double* f;
	double* pbest_x;
	double* pbest_f;
	double* prev_pbest_x;
	double* prev_pbest_f;
} Swarm;

double himmelblau(double* x) {
	double x1 = x[0];
	double x2 = x[1];
	return (x1 * x1 + x2 - 11) * (x1 * x1 + x2 - 11) + (x1 + x2 * x2 - 7) * (x1 + x2 * x2 - 7);
}

static double* ref_score = NULL;

int sort_elites(const void* a, const void* b) {
	int idx_a = *(const int*) a;
	int idx_b = *(const int*) b;

	if (ref_score[idx_a] < ref_score[idx_b]) return -1;
	if (ref_score[idx_a] > ref_score[idx_b]) return 1;
	return 0;
}

void pso_run(int swarm_size, int max_iter) {

	// ---[ one off stuff ]-----------------------------------------
	int dim = 2;
	double max_x_perturbation = 5;
	double max_v_perturbation = 0.1;
	size_t vector_size = dim * sizeof(double);
	int eliten = 6;
	int is_elite[swarm_size], indices[swarm_size];
	for (int p = 0; p < swarm_size; p++) {
		is_elite[p] = 0;
		indices[p] = p;
	}

	// ===[ INIT SWARM ]============================================
	size_t memblock = sizeof(Swarm) + (4 * swarm_size * dim + 3 * swarm_size) * sizeof(double);
	Swarm* swarm = malloc(memblock);
	if (!swarm) exit(1);

	double* mem = (double*) (swarm + 1);
	swarm->x = mem; mem += swarm_size * dim;
	swarm->v = mem; mem += swarm_size * dim;
	swarm->f = mem; mem += swarm_size;
	swarm->pbest_x = mem; mem += swarm_size * dim;
	swarm->pbest_f = mem; mem += swarm_size;
	swarm->prev_pbest_x = mem; mem += swarm_size * dim;
	swarm->prev_pbest_f = mem; mem += swarm_size;

	// create a seed for each particle bc rand is not thread safe
	unsigned int swarm_seeds[swarm_size];
	for (int p = 0; p < swarm_size; p++)
		swarm_seeds[p] = (unsigned int) rand();


	// ---[ parallel init ]-----------------------------------------
	#pragma omp parallel for schedule(guided, 2)
	for (int p = 0; p < swarm_size; p++) {

		double* x_p = &swarm->x[p * dim];
		double* v_p = &swarm->v[p * dim];
		double* f_p = &swarm->f[p];
		double* pbest_x_p = &swarm->pbest_x[p * dim];
		double* pbest_f_p = &swarm->pbest_f[p];
		double* prev_pbest_x_p = &swarm->prev_pbest_x[p * dim];
		double* prev_pbest_f_p = &swarm->prev_pbest_f[p];

		for (int d = 0; d < dim; d++) {
			x_p[d] = ((double) rand_r(&swarm_seeds[p]) / RAND_MAX) * 2 * max_x_perturbation - max_x_perturbation;
			v_p[d] = ((double) rand_r(&swarm_seeds[p]) / RAND_MAX) * 2 * max_v_perturbation - max_v_perturbation;
		}

		*f_p = himmelblau(x_p);

		memcpy(pbest_x_p, x_p, vector_size);
		memcpy(prev_pbest_x_p, x_p, vector_size);
		*pbest_f_p = *f_p;
		*prev_pbest_f_p = *f_p;
	}
	// ---[ serial elites init ]------------------------------------
	ref_score = swarm->pbest_f;
	qsort(indices, swarm_size, sizeof(int), sort_elites);
	for (int k = 0; k < eliten; k++)
		is_elite[indices[k]] = 1;

	// =============================================================
	


	// =============================================================
	// ===[ PSO OPT RUN ]===========================================
	// =============================================================


	// ---[ pso params ]--------------------------------------------
	double w = 0.729, c1 = 1.49445, c2 = 1.49445;
	double v_max = 6;
	// -------------------------------------------------------------

	for (int iter = 0; iter < max_iter; iter++) {

		// ---[ parallel run ]-----------------------------------------------
		#pragma omp parallel for schedule(guided, 2)
		for (int p = 0; p < swarm_size; p++) {

			int next_particle_idx = (p + 1) % swarm_size;

			// prev_pbest is read only while pbest is write only so i dont bother with omp locks.
			// this means that this time velocity updates are not asynchronous and info travels slower along the swarm
			double* x_p = &swarm->x[p * dim];
			double* v_p = &swarm->v[p * dim];
			double* f_p = &swarm->f[p];
			double* pbest_x_p = &swarm->pbest_x[p * dim];
			double* pbest_f_p = &swarm->pbest_f[p];
			double* prev_pbest_x_p = &swarm->prev_pbest_x[p * dim];
			double* prev_pbest_f_p = &swarm->prev_pbest_f[p];
			double* lbest_x, lbest_f;

			// lbests are not stored, just a temp pointer
			double* neigh_pbest_x = &swarm->prev_pbest_x[next_particle_idx * dim];
			double* neigh_pbest_f = &swarm->prev_pbest_f[next_particle_idx];
			if (*neigh_pbest_f < *prev_pbest_f_p) {
				lbest_x = neigh_pbest_x;
				lbest_f = *neigh_pbest_f;
			} else {
				lbest_x = prev_pbest_x_p;
				lbest_f = *prev_pbest_f_p;
			}

			unsigned int* seed = &swarm_seeds[p];
			double local_c2 = (is_elite[next_particle_idx]) ? 0 : c2;

			for (int d = 0; d < dim; d++) {
				double r1 = (double) rand_r(seed) / RAND_MAX;
				double r2 = (double) rand_r(seed) / RAND_MAX;

				v_p[d] *= w;
				v_p[d] += c1 * r1 * (prev_pbest_x_p[d] - x_p[d]);
				v_p[d] += local_c2 * r2 * (lbest_x[d] - x_p[d]);

				v_p[d] = (v_p[d] > v_max) ? v_max : v_p[d];
				v_p[d] = (v_p[d] < -v_max) ? -v_max : v_p[d];

				x_p[d] += v_p[d];
			}

			*f_p = himmelblau(x_p);

			if (*f_p < *pbest_f_p) {
				*pbest_f_p = *f_p;
				memcpy(pbest_x_p, x_p, vector_size);
			}
		}
		// ---[ serial jobs ]-------------------------------------------

		// update pbest of prev iter
		memcpy(swarm->prev_pbest_x, swarm->pbest_x, swarm_size * dim * sizeof(double));
		memcpy(swarm->prev_pbest_f, swarm->pbest_f, swarm_size * sizeof(double));
		for (int p = 0; p < swarm_size; p++)
			is_elite[p] = 0;

		// update elites arr
		qsort(indices, swarm_size, sizeof(int), sort_elites);
		for (int k = 0; k < eliten; k++)
			is_elite[indices[k]] = 1;
	}
	// =============================================================
	// ===[ PSO OPT RUN END ]=======================================
	// =============================================================

	for (int k = 0; k < eliten; k++) {
		printf("rank %d: %f, (%f, %f)\n", k + 1, swarm->pbest_f[indices[k]], swarm->pbest_x[indices[k] * dim], swarm->pbest_x[indices[k] * dim + 1]);
	}

	free(swarm);
}

int main(void) {

	int swarm_size = 40;
	int max_iter = 2000;

	pso_run(swarm_size, max_iter);

	return 0;
}
