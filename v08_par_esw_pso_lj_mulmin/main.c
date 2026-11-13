#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <omp.h>
#include "main.h"
#include "read_write.h"
#include "pso_run.h"

void free_atoms(Atoms* atoms) {
	free(atoms->coords);
	for (int i = 0; i < atoms->atomn; i++)
		free(atoms->types[i]);
	free(atoms->types);
	free(atoms);
}

int main(void) {

	mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
	omp_set_num_threads(8);

	char* filename = "Ar7_LJ.xyz";

	Atoms* atoms = malloc(sizeof(Atoms));
	read_in(filename, atoms);

	int swarm_size = 40;
	int max_iter = 4000;

	Swarm* swarm = pso_run(atoms, swarm_size, max_iter);

	free_atoms(atoms);
	free(swarm);

	return 0;
}
