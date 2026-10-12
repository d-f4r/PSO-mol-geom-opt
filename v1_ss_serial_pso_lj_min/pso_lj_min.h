#ifndef PSO_LJ_MIN_H
#define PSO_LJ_MIN_H

#include "main.h"

typedef struct {
	double* x;
	double* v;
	double* pbest_x;
	double* pbest_value;
	double* gbest_x;
	double gbest_value;
} Swarm;

Swarm* pso_min_run(Atoms* atoms, int swarm_size, int max_iter);

#endif
