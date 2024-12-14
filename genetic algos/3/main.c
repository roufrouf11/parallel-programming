#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#include "main.h"

int genome_length;
char *perfect;

char GENES[26] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

int main() {
    Population *pops, pop;
    double cost = 0;
    int ready = 0;
    int done = 0;
    Organism shared[POPULATION_COUNT][SHARE_COUNT];
    FILE *file = fopen("12.txt", "w");

    srand(time(NULL));

    // Manually set the number of threads for testing
    omp_set_dynamic(0);
    omp_set_num_threads(THREADS_COUNT);

    for (genome_length = 10; genome_length <= 50; genome_length += 10) {
        cost = 0;
        for (int w = 0; w < ITERATIONS; w++) {
            done = 0;
            ready = 0;

            create_perfect();
            pops = initialize_populations();

            cost -= omp_get_wtime();
            while (!done) {
                #pragma omp parallel private(pop)
                {
                    unsigned int seed = omp_get_thread_num();
                    #pragma omp for 
                    for (int i = 0; i < POPULATION_COUNT; i++) {
                        pop = pops[i];

                        for (int k = 0; k < SHARE_EVERY - 1; k++) {
                            if (pop.max_fitness == genome_length) {
                                done = 1;
                                break;
                            }

                            reproduce(&pop, &seed, 0);
                            evaluate_population(&pop);
                        }
                        pops[i] = pop;
                    }

                    // Share organisms between populations
                    for (int i = 0; i < POPULATION_COUNT; i++) {
                        for (int k = 0; k < SHARE_COUNT; k++) {
                            shared[i][k] = pop.organisms[roulette_select(pop, &seed)];
                        }
                    }
                }
                if (done) break;
                #pragma omp parallel private (pop)
                {
                    unsigned int seed = omp_get_thread_num();
                    #pragma omp for
                    for (int i = 0; i < POPULATION_COUNT; i++) {
                        pop = pops[i];
                        reproduce(&pop, &seed, 1);
                        int random_count = ORGANISMS_PER_POP - SHARE_COUNT * POPULATION_COUNT;
                        for (int i = 0; i < ORGANISMS_PER_POP - random_count; i++) {
                            pop.organisms[i] = shared[i / POPULATION_COUNT][i % POPULATION_COUNT];
                            pop.organisms[i].genome = strndup(pop.organisms[i].genome, genome_length);
                        }
                        evaluate_population(&pop);
                        pops[i] = pop;
                    }
                }
                
            }
            cost += omp_get_wtime();
        }
        printf("Average time for genome length = %d: %lf\n", genome_length, cost / ITERATIONS);
        fprintf(file, "%d, %lf\n", genome_length, cost / ITERATIONS);

        free(perfect);
    }
    fclose(file);
    return 0;
}

void create_perfect() {
    perfect = malloc(sizeof(char) * genome_length + 1);
    for (int i = 0; i < genome_length; i++) {
        perfect[i] = GENES[rand() % ALLELES_COUNT];
    }
    perfect[genome_length] = 0;
}

Population *initialize_populations() {
    Population *pops = malloc(sizeof(Population) * POPULATION_COUNT);

    for (int i = 0; i < POPULATION_COUNT; i++) {
        // Randomize initial genome
        for (int j = 0; j < ORGANISMS_PER_POP; j++) {
            pops[i].organisms[j].genome = malloc(sizeof(char) * genome_length + 1);
            for (int k = 0; k < genome_length; k++) {
                pops[i].organisms[j].genome[k] = GENES[rand() % ALLELES_COUNT];
            }
            pops[i].organisms[j].genome[genome_length] = 0;
        }

        evaluate_population(&pops[i]);
    }

    return pops;
}

void evaluate_population(Population *pop) {
    // Calculates the fitness of each organism of the population
    pop->total_fitness = 0;
    pop->max_fitness = 0;
    for (int i = 0; i < ORGANISMS_PER_POP; i++) {
        pop->organisms[i].fitness = 0;
        for (int j = 0; j < genome_length; j++) {
            if (pop->organisms[i].genome[j] == perfect[j]) {
                pop->organisms[i].fitness++;
            }
        }
        pop->total_fitness += pop->organisms[i].fitness;

        if (pop->organisms[i].fitness > pop->max_fitness) {
            pop->max_fitness = pop->organisms[i].fitness;
        }
    }
}

int roulette_select(Population pop, unsigned int *seed) {
    // Returns the index of the selected organism
    int threshold = rand_r(seed) % (pop.total_fitness + 1);
    int running_total = 0;

    for (int i = 0; i < ORGANISMS_PER_POP; i++) {
        running_total += pop.organisms[i].fitness;
        if (running_total > threshold) {
            return i;
        }
    }
    return 0;
}

void reproduce(Population *pop, unsigned int *seed, int use_shared) {
    Population pop_copy = *pop;
    int a, b;
    int split;
    int random_count;

    // Deep copy pop
    for (int i = 0; i < ORGANISMS_PER_POP; i++) {
        pop_copy.organisms[i].genome = strndup(pop->organisms[i].genome, genome_length);
    }
    
    if (use_shared) {
        random_count = ORGANISMS_PER_POP - SHARE_COUNT * POPULATION_COUNT;
    } else {
        random_count = ORGANISMS_PER_POP;
    }

    for (int i = 0; i < random_count; i++) {
        a = roulette_select(pop_copy, seed);
        b = roulette_select(pop_copy, seed);

        // Merge genomes from parents
        split = rand_r(seed) % (genome_length + 1);
        for (int j = 0; j < split; j++) {
            pop->organisms[i].genome[j] = pop_copy.organisms[a].genome[j];
        }
        for (int j = split; j <= genome_length; j++) {
            pop->organisms[i].genome[j] = pop_copy.organisms[b].genome[j];
        }

        // Mutate
        for (int j = 0; j < genome_length; j++) {
            if ((rand_r(seed) % 10000) / 10000.0f < MUTATION_RATE) {
                pop->organisms[i].genome[j] = GENES[rand_r(seed) % ALLELES_COUNT];
            }
        }
    }

    pop->generation++;

    for (int i = 0; i < ORGANISMS_PER_POP; i++) {
        free(pop_copy.organisms[i].genome);
    }
}
