#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#include "main.h"

char *perfect;

char GENES[26] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

int genome_length;
int total_fitness = 0;
int max_fitness = 0;
int generation = 0;

int main() {
    Organism *pop;
    FILE *file = fopen("12.txt", "w");
    double cost;

    srand(time(NULL));

    // Manually set the number of threads for testing
    omp_set_dynamic(0);
    omp_set_num_threads(THREADS_COUNT);

    for (genome_length = 10; genome_length <= 50; genome_length += 10) {
        cost = 0;

        for (int w = 0; w < ITERATIONS; w++) {
            create_perfect();
            pop = initialize_populations();

            cost -= omp_get_wtime();
            while (1)  {
                if (max_fitness == genome_length) break;

                pop = reproduce(pop);
                evaluate_population(pop);
            }
            cost += omp_get_wtime();
            free(perfect);
        }
        printf("Average time for genome length = %d: %lf\n",
               genome_length, cost / ITERATIONS);
        fprintf(file, "%d, %lf, %d\n", genome_length, cost / ITERATIONS, generation);
    }
    fclose(file);
    return 0;
}

void create_perfect() {
    perfect = malloc(sizeof(char) * genome_length);
    for (int i = 0; i < genome_length; i++) {
        perfect[i] = GENES[rand() % ALLELES_COUNT];
    }
}

Organism *initialize_populations() {
    Organism *pop = aligned_alloc(64, sizeof(Organism) * ORGANISMS_PER_POP);

    // Randomize initial genome
    for (int j = 0; j < ORGANISMS_PER_POP; j++) {
        pop[j].genome = malloc(sizeof(char) * genome_length);
        for (int k = 0; k < genome_length; k++) {
            pop[j].genome[k] = GENES[rand() % ALLELES_COUNT];
        }
    }

    evaluate_population(pop);

    return pop;
}

void evaluate_population(Organism *pop) {
    // Calculates the fitness of each organism of the population
    int fitness_arr[ORGANISMS_PER_POP];
    int temp_fit;
    #pragma omp parallel for private (temp_fit)
    for (int i = 0; i < ORGANISMS_PER_POP; i++) {
        temp_fit = 0;
        for (int j = 0; j < genome_length; j++) {
            if (pop[i].genome[j] == perfect[j]) {
                temp_fit++;
            }
        }

        pop[i].fitness = temp_fit;
        fitness_arr[i] = temp_fit;
    }

    total_fitness = 0;
    max_fitness = 0;
    #pragma omp parallel for reduction (+:total_fitness) reduction (max:max_fitness) 
    for (int i = 0; i < ORGANISMS_PER_POP; i++) {
        total_fitness += fitness_arr[i];
        max_fitness = max_fitness > fitness_arr[i] ? max_fitness : fitness_arr[i];
    }
}

int roulette_select(Organism *pop, unsigned int *seed) {
    // Returns the index of the selected organism
    int threshold = rand_r(seed) % (total_fitness + 1);
    int running_total = 0;

    for (int i = 0; i < ORGANISMS_PER_POP; i++) {
        running_total += pop[i].fitness;
        if (running_total > threshold) {
            return i;
        }
    }
    return 0;
}

Organism *reproduce(Organism *pop) {
    Organism *new_pop ;
    int a, b;
    int split;
    char new_genome[genome_length];

    new_pop = aligned_alloc(64, sizeof(Organism) * ORGANISMS_PER_POP);

    #pragma omp parallel private (a, b, split, new_genome)
    {
        unsigned int seed = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < ORGANISMS_PER_POP; i++) {
            a = roulette_select(pop, &seed);
            b = roulette_select(pop, &seed);
            // Merge genomes from parents
            split = rand_r(&seed) % (genome_length + 1);
            for (int j = 0; j < split; j++) {
                new_genome[j] = pop[a].genome[j];
            }
            
            for (int j = split; j < genome_length; j++) {
                new_genome[j] = pop[b].genome[j];
            }
            // Mutate
            for (int j = 0; j < genome_length; j++) {
                if ((rand_r(&seed) % 10000) / 10000.0f < MUTATION_RATE) {
                    new_genome[j] = GENES[rand_r(&seed) % ALLELES_COUNT];
                }
            }
            new_pop[i].genome = strdup(new_genome);
        }
    }
    generation++;
    for (int i = 0; i < ORGANISMS_PER_POP; i++) {
        free(pop[i].genome);
    }
    free(pop);
    return new_pop;
}
