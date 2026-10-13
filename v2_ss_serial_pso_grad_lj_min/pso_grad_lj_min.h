#ifndef PSO_GRAD_LJ_MIN_H
#define PSO_GRAD_LJ_MIN_H

#include "main.h"

typedef struct {
	double* x;
	double* v;
	double* g;
	double* pbest_x;
	double* pbest_value;
	double* g_norm_sq;
	double* gbest_x;
	double* gbest_g;
	double gbest_value;
} Swarm;

Swarm* pso_grad_min_run(Atoms* atoms, int swarm_size, int max_iter);

#endif
