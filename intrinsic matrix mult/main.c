#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <immintrin.h>
#include <math.h>

int ITERATIONS_NUM = 25;

int main(int argc, char **argv)
{
    double **A, **B, **C, **B_T, **C_T;
    __m256d b;
    int M, N, W, padded_N;
    double cost, result;
    FILE *file = fopen("12_2.txt", "w");
    __m256d *partial_sums;
    size_t vector_size = sizeof(__m256d);
    srand(time(NULL));

    // Manually set the number of threads for testing
    omp_set_dynamic(0);
    omp_set_num_threads(12);
    
    for (int size = 11; size <= 20; size++) {
        M = 100 * size;
        N = 100 * size;
        W = 100 * size;
        cost = 0;

        // Calculate matrix size after padding
        padded_N = N + (N % 4 ? 4 - N % 4 : 0);

        for (int epoch = 0; epoch < ITERATIONS_NUM; epoch++) {
            // Fill A
            A = malloc(sizeof(double *) * M);
            for (int i = 0; i < M; i++)
            {
                A[i] = malloc(sizeof(double) * padded_N);
                for (int j = 0; j < N; j++)
                {
                    A[i][j] = rand() % 3;
                }
                for (int j = N; j < padded_N; j++)
                {
                    A[i][j] = 0;
                }
            }

            // Fill B
            B = malloc(sizeof(double *) * padded_N);
            for (int i = 0; i < N; i++)
            {
                B[i] = malloc(sizeof(double) * W);
                for (int j = 0; j < W; j++)
                {
                    B[i][j] = rand() % 3;
                }
            }
            for (int i = N; i < padded_N; i++)
            {
                B[i] = malloc(sizeof(double) * W);
                for (int j = 0; j < W; j++)
                {
                    B[i][j] = 0;
                }
            }

            // Malloc for C
            C = malloc(sizeof(double *) * M);
            for (int i = 0; i < M; i++)
            {
                C[i] = malloc(sizeof(double) * W);
            }

            // Malloc for C_T
            C_T = malloc(sizeof(double *) * W);
            for (int i = 0; i < W; i++)
            {
                C_T[i] = aligned_alloc(64, ceil((double) sizeof(double) * M / 64) * 64);
            }

            // Calculate B_T (B transposed; we need this to load row vectors)
            B_T = malloc(sizeof(double *) * W);
            for (int i = 0; i < W; i++)
            {
                B_T[i] = malloc(sizeof(double) * padded_N);
                for (int j = 0; j < padded_N; j++)
                {
                    B_T[i][j] = B[j][i];
                }
            }
            cost -= omp_get_wtime();
            // Fill C
            #pragma omp parallel for private (b, result, partial_sums)
            for (int j = 0; j < W; j++)
            {
                #pragma omp task
                {
                    if (
                        argc > 1 &&
                        (!strcmp(argv[1], "--verbose=1") || !strcmp(argv[1], "--verbose=2"))
                    ) {
                        printf("Process %d received task %d\n", omp_get_thread_num(), j);
                    }

                    partial_sums = aligned_alloc(vector_size, vector_size * M);
                    for (int i = 0; i < M; i++) {
                        partial_sums[i] = _mm256_setzero_pd();
                    }

                    for (int k = 0; k < M; k += 4) {
                        b = _mm256_loadu_pd(&B_T[j][k]);
                        for (int i = 0; i < N; i += 1) {
                            partial_sums[i] = _mm256_add_pd(
                                partial_sums[i],
                                _mm256_mul_pd(
                                    _mm256_loadu_pd(&A[i][k]), b
                                )
                            );
                        }
                    }

                    for (int i = 0; i < M; i++) {
                        result = 0;
                        for (int l = 0; l < 4; l++) {
                            result += partial_sums[i][l];
                        }
                        // Fill the transposde array to avoid false sharing
                        C_T[j][i] += result;
                    }
                    free(partial_sums);
                }
            }

            cost += omp_get_wtime();

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < W; j++) {
                    C[i][j] = C_T[j][i];
                }
            }

            // Print matrices
            if (argc > 1 && !strcmp(argv[1], "--verbose=2")) {
                printf("A:\n");
                printf("[");
                for (int i = 0; i < M; i++)
                {
                    printf("[");
                    for (int j = 0; j < N; j++)
                    {
                        if (j < N - 1) printf("%lf, ", A[i][j]);
                        else printf("%lf]", A[i][j]);
                    }
                    if (i < M - 1) printf(",\n");
                    else printf("]\n");
                }

                printf("\nB:\n");
                printf("[");
                for (int i = 0; i < N; i++)
                {
                    printf("[");
                    for (int j = 0; j < W; j++)
                    {
                        if (j < W - 1) printf("%lf, ", B[i][j]);
                        else printf("%lf]", B[i][j]);
                    }
                    if (i < N - 1) printf(",\n");
                    else printf("]\n");
                }

                printf("\nC:\n");
                printf("[");
                for (int i = 0; i < M; i++)
                {
                    printf("[");
                    for (int j = 0; j < W; j++)
                    {
                        if (j < W - 1) printf("%lf, ", C[i][j]);
                        else printf("%lf]", C[i][j]);
                    }
                    if (i < M - 1) printf(",\n");
                    else printf("]\n");
                }
            }

            for (int i = 0; i < M; i++) {
                free(A[i]);
                free(C[i]);
            }
            free(A);
            free(C);
            for (int i = 0; i < padded_N; i++) {
                free(B[i]);
            }
            free(B);
            for (int i = 0; i < W; i++) {
                free(B_T[i]);
                free(C_T[i]);
            }
            free(B_T);
            free(C_T);
        }

        printf("The average time of matrix multiplication for %dx%d matrices was %lf seconds\n",
            M, W, cost / ITERATIONS_NUM);
        fprintf(file, "%d, %lf\n", size, cost / ITERATIONS_NUM);
    }
    fclose(file);

    return 0;
}
