#ifndef PSO_RUN_H
#define PSO_RUN_H

#include "main.h"

typedef struct {
	double* x;
	double* v;
	double* g;
	double* pbest_x;
	double* pbest_g;
	double* pbest_e;
	double* pbest_f;
	double* pbest_g_norm_sq;
	double* prev_iter_pbest_x;
	double* prev_iter_pbest_e;
	double* prev_iter_pbest_f;
	double* prev_iter_pbest_g_norm_sq;
} Swarm;

Swarm* pso_run(Atoms* atoms, int sub_swarm_size, int max_iter);

#endif
