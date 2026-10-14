#ifndef READ_WRITE_H
#define READ_WRITE_H

#include "main.h"
#include "pso_run.h"

void read_in(char* filename, Atoms* atoms);
void print_final_geom(char* filename, Swarm* swarm, Atoms* atoms);

#endif
