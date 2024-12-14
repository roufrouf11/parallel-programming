#ifndef MAIN_H
#define MAIN_H

#define ITERATIONS 50
#define ALLELES_COUNT 4
#define POPULATION_COUNT 12
#define ORGANISMS_PER_POP 1000
#define MUTATION_RATE 0.001
#define THREADS_COUNT 12

struct Organism {
    int fitness;
    char *genome;
} typedef Organism;

struct Population {
    int max_fitness;
    int total_fitness;
    int generation;
    Organism organisms[ORGANISMS_PER_POP];
} typedef Population;

void create_perfect();

Population *initialize_populations();

void evaluate_population(Population *pop);

int roulette_select(Population pop, unsigned int *seed);

void reproduce(Population *pop, unsigned int *seed);

#endif