#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <math.h>

int ITERATIONS_NUM = 50;

int main(int argc, char **argv)
{
    double **A, **B, **C;
    int M, N, W;
    double cost = 0;
    FILE *file = fopen("12.txt", "w");
    srand(time(NULL));

    // Manually set the number of threads for testing
    omp_set_dynamic(0);
    omp_set_num_threads(12);

    for (int size = 1; size <= 10; size++) {
        N = 100 * size;
        M = 100 * size;
        W = 100 * size;
        cost = 0;

        for (int epoch = 0; epoch < ITERATIONS_NUM; epoch++) {
            // Fill A
            A = malloc(sizeof(double *) * M);
            for (int i = 0; i < M; i++)
            {
                A[i] = malloc(sizeof(double) * N);
                for (int j = 0; j < N; j++)
                {
                    A[i][j] = rand() % 3 / (rand() % 10 + 1);
                }
            }

            // Fill B
            B = malloc(sizeof(double *) * N);
            for (int i = 0; i < N; i++)
            {
                B[i] = malloc(sizeof(double) * W);
                for (int j = 0; j < W; j++)
                {
                    B[i][j] = rand() % 3 / (rand() % 10 + 1);
                }
            }

            // Malloc for C
            C = malloc(sizeof(double *) * M);
            for (int i = 0; i < M; i++)
            {
                C[i] = malloc(sizeof(double) * W);
            }

            double **C_T = malloc(sizeof(double *) * W);
            for (int i = 0; i < W; i++)
            {
                C_T[i] = aligned_alloc(64, ceil((double) sizeof(double) * M / 64) * 64);
            }

            // Fill C
            cost -= omp_get_wtime();
            #pragma omp parallel for
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
                    for (int i = 0; i < M; i++)
                    {
                        double result = 0;
                        for (int k = 0; k < N; k++) 
                        {
                            result += A[i][k] * B[k][j];
                        }
                        C_T[j][i] = result;
                    }
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
            for (int i = 0; i < N; i++) {
                free(B[i]);
            }
            free(B);
            for (int i = 0; i < W; i++) {
                free(C_T[i]);
            }
            free(C_T);
        }

        printf("The average time of matrix multiplication for %dx%d matrices was %lf seconds\n",
            M, W, cost / ITERATIONS_NUM);
        fprintf(file, "%d, %lf\n", size, cost / ITERATIONS_NUM);
    }
    fclose(file);
    return 0;
}
