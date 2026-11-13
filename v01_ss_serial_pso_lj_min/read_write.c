#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"
#include "pso_lj_min.h"


#define BUFFER_SIZE 512

char** strsplit(char* string_to_split) {

	char **token_arr = malloc(4 * sizeof(char *));
	if (!token_arr) {
		fprintf(stderr, "cant malloc token array\n");
		exit(EXIT_FAILURE);
	}

	const char *delimiter = " \n";
	int token_count = 0;

	char *token = strtok(string_to_split, delimiter);

	while (token != NULL && token_count < 4) {
		token_arr[token_count] = strdup(token); // prepei na ka8aristei auto meta
		token = strtok(NULL, delimiter);
		token_count++;
	}

	if (token_count < 4) {
		fprintf(stderr, "found %d tokens\n", token_count);
		for (int i = 0; i < token_count; i++)
			free(token_arr[i]);
		free(token_arr);
		exit(EXIT_FAILURE);
	}

	return token_arr;
}

void read_in(char* filename, Atoms* atoms) {
	FILE* rstream = fopen(filename, "r"); 
	if (rstream == NULL) {
		fprintf(stderr, "cant open file\n");
		exit(1);
	}

	char line_buffer[BUFFER_SIZE];
	int atomn = atoi(fgets(line_buffer, sizeof(line_buffer), rstream));

	atoms->atomn = atomn;
	atoms->coords = malloc(3 * atomn * sizeof(double));
	atoms->types = malloc(atomn * sizeof(char*));

	fgets(line_buffer, sizeof(line_buffer), rstream);


	int current_atom = 0;
	while (fgets(line_buffer, sizeof(line_buffer), rstream)) {
		char** token_arr = strsplit(line_buffer);

		atoms->types[current_atom] = token_arr[0];
		atoms->coords[3 * current_atom] = atof(token_arr[1]);
		atoms->coords[3 * current_atom + 1] = atof(token_arr[2]);
		atoms->coords[3 * current_atom + 2] = atof(token_arr[3]);

		for (int i = 1; i < 4; i++)
			free(token_arr[i]);
		free(token_arr);
		current_atom++;
	}
	fclose(rstream);
}

void print_final_geom(char* filename, Swarm* swarm, Atoms* atoms) {

	int filename_buffer = 512;
	char new_filename[filename_buffer];
	int i = 0;
	while (filename[i] != '\0' && filename[i] != '.') {
		new_filename[i] = filename[i];
		i++;
	}
	strcat(new_filename, "_out.gjf");

	FILE* wstream = fopen(new_filename, "w"); 
	if (wstream == NULL) {
		fprintf(stderr, "cant open write file\n");
		exit(1);
	}

	int atomn = atoms->atomn;
	fprintf(wstream, "# hf/3-21g\n\n--\n\n0 1\n");
	for (int i = 0; i < atomn; i++) {
		fprintf(wstream, "%s\t%12.8f\t%12.8f\t%12.8f\n", atoms->types[i], swarm->gbest_x[3 * i], swarm->gbest_x[3 * i + 1], swarm->gbest_x[3 * i + 2]);
	}
	fprintf(wstream, "\n\n");
	fclose(wstream);
}

