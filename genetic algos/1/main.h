#ifndef MAIN_H
#define MAIN_H

#define ITERATIONS 50
#define ALLELES_COUNT 4
#define ORGANISMS_PER_POP 5000
#define MUTATION_RATE 0.001
#define THREADS_COUNT 12

struct Organism {
    int fitness;
    char *genome;
} __attribute__((packed,aligned(64))) typedef Organism;

void create_perfect();

Organism *initialize_populations();

void evaluate_population(Organism *pop);

int roulette_select(Organism *pop, unsigned int *seed);

Organism *reproduce(Organism *pop);

#endif