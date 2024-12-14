# Parallel Programming Systems 

## Overview
This project explores techniques to optimize parallel programming performance, including memory alignment, compiler optimizations, intrinsic functions, and the implementation of a genetic algorithm. The experiments are designed to analyze and improve computation efficiency on multi-threaded systems.

## Tasks

### Task A: Basic Multiplication
- **Objective**: Perform parallelized matrix multiplication while considering memory alignment to avoid false sharing.
- **Optimization**:
  - Applied `O3` optimizations in `gcc`.
  - Benchmarked performance for optimized vs. unoptimized code.
- **Results**:
  - Optimized code significantly outperforms unoptimized code, even with a single thread.
  - Observed bottlenecks in memory bandwidth when scaling to 12 threads compared to 6 threads, due to increased memory fetch operations.

### Task B: Intrinsics
- **Objective**: Compare performance with and without intrinsic functions.
- **Optimization**:
  - Leveraged SIMD intrinsic functions to improve computation efficiency.
- **Results**:
  - Substantial performance gains observed with intrinsic functions.
  - Performance scales well with increased thread count and larger matrix sizes.

### Task C: Genetic Algorithm
- **Objective**: Implement a genetic algorithm to solve optimization problems.
- **Subtasks**:
  1. Initialization and population generation.
  2. Fitness evaluation and selection.
  3. Crossover and mutation for population evolution.
- **Results**:
  - Detailed analysis and benchmarking provided for each subtask.

## Performance Highlights
- **Task A**: Optimized matrix multiplication scales effectively but exhibits memory bandwidth limitations beyond a certain thread count.
- **Task B**: Intrinsics provide significant speedups, demonstrating the importance of low-level optimization in high-performance computing.
- **Task C**: Genetic algorithm successfully implemented with detailed performance metrics.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```
2. Compile the code:
   ```bash
   gcc -O3 -fopenmp taskA.c -o taskA
   gcc -O3 -fopenmp taskB.c -o taskB
   gcc -O3 -fopenmp taskC.c -o taskC
   ```

## Usage
1. **Task A**: Run the matrix multiplication benchmark:
   ```bash
   ./taskA
   ```
2. **Task B**: Run the intrinsics performance comparison:
   ```bash
   ./taskB
   ```
3. **Task C**: Execute the genetic algorithm:
   ```bash
   ./taskC
   ```

## Observations
- Memory alignment and `O3` optimizations are critical for mitigating performance issues like false sharing.
- Intrinsics accelerate computational tasks by leveraging SIMD capabilities of modern CPUs.
- Genetic algorithms are effective for optimization tasks but require careful tuning for performance.

## File Structure
- `taskA.c`: Source code for matrix multiplication.
- `taskB.c`: Source code for intrinsics performance analysis.
- `taskC.c`: Source code for genetic algorithm implementation.
- `README.md`: Project documentation.

## Acknowledgments
This project was developed as part of the **Parallel Programming Systems** course.

