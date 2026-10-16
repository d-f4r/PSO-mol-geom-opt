#include <stdio.h>
#include <stdlib.h>
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

	char* filename = "Ar7_LJ.xyz";

	Atoms* atoms = malloc(sizeof(Atoms));
	read_in(filename, atoms);

	int sub_swarm_n = 4;
	int swarm_size = 80;
	int max_iter = 5000;

	Swarm* swarm = pso_run(atoms, sub_swarm_n, swarm_size, max_iter);

	// print_final_geom(filename, swarm, atoms);

	free_atoms(atoms);
	free(swarm);

	return 0;
}
