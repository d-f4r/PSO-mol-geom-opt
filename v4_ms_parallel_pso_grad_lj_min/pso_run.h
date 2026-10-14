#ifndef PSO_RUN_H
#define PSO_RUN_H

#include "main.h"

typedef struct {
	double* lbest_x;
	double* lbest_g;
	double* lbest_e;
	double* gbest_x;
	double* gbest_g;
	double gbest_e;
} Swarm;

typedef struct {
	double* x;
	double* v;
	double* g;
	double* pbest_x;
	double* pbest_g;
	double* pbest_e;
	double* g_norm_sq;
} SubSwarms;

Swarm* pso_run(Atoms* atoms, int sub_swarm_n, int sub_swarm_size, int max_iter);

#endif
