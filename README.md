# Department-Rank-Improver
Optimisation Techniques to improve the ranking of department in terms of research papers, publications and other miscellaneous stuff

## Optimisation Techniques and Parallel Computing
This project aims to suggest a hybrid algorithm to perform Multi Objective Optimisation to increase the department rankings based on the research papers published and shown at major tier-1 conferences. We define the following three fitness functions:
1. maximise $O_1(x) = \cfrac{\text{papers}_{\texttt{T1}}}{\text{research}_{\texttt{FTE}}}$

2. maximise $O_2(x) = \cfrac{\sum_{i}w_i\cdot \texttt{cite}\%}{\text{research}_{\texttt{FTE}}}$

3. minimise $O_3(x) = \cfrac{\text{research budget}}{O_2(x) + \epsilon}$

The optimiser must explore feasible solutions that satisfy constraints such as:
* Minimum faculty strength
* Valid research time fractions
* Reasonable student-faculty ratios
* Non-negative budgets

The result is a set of optimised departmental configurations that maximise research impact while minimizing cost.

### Approach
The RION (Research Innovation Societal Optimisation Network) is a hybrid multi-objective optimisation framework that integrates NSGA-II with a PSO-based mutation mechanism. This approach enhances the global search (via NSGA-II) while respecting constraints defined on each variable.

1. Initialisation: Generate a population by sampling randomly from the dataset.
2. NSGA-II Evolution Loop: Each generation performs:

    1. Parent Selection and Crossover: Tournament selection is used and crossover is a simple random chance to swap the values of a single objective of the parents.
    2. Population merging: Parents and offspring are now combined into a single population.
    3. Fast-non dominated sorting: Sort the new population into pareto fronts
    4. Crowding distance calculations and next generation selection: Fronts are added in order until population limit is reached. If the final front overflows, its individuals are sorted by crowding distance to select the most diverse solutions.
3. PSO-Driven mutation: Instead of classical random mutation, RION introduces targeted PSO optimisation:
    * One attribute (column) is randomly chosen.
    * All offspring values in that column act as particles in a PSO swarm.
    * PSO updates particle positions for a fixed number of iterations.

    PSO Objective function: $$\texttt{PSO\_fitness}(x) = \sum_{i = 1}^3O_i(x) + p\cdot\texttt{cv}(x)$$
    where $p$ is the penalty for violating constraints (scaling factor) and $\texttt{cv}(x)$ is the constraint violation function which just is the fraction of the budget which is violated. $$\texttt{cv}(x) = \sum_{a\,\in\,x.columns} \min\left(0, \cfrac{minv[a] - x[a]}{|minv[a]|}\right) + \max\left(0, \cfrac{x[a] - maxv[a]}{|maxv[a]|}\right)$$

4. Final output: After all generations:
    * The final population contains non-dominated, diverse, constraint-satisfying solutions.
    * These solutions form an approximate Pareto front optimised using both evolutionary and swarm based intelligence principles.

## Parallelism 

The Particle Swarm mutation step is implemented using a Parallel Approach. For any given iteration of the exploration step, all the particles are independent of each other and can effectively carry out the calculations without needing to wait for each other.

Since we just require basic for-loop unrolling for the parallelisation, we make use of the `OpenMP` API for the C programming language. To call the parallel optimisation function written in C we make use of an interface called `ctypes` in our Python code. The following is the main driving part of the Python code which calls the optimisation function written in C.

```py
import ctypes
from ctypes import CFUNCTYPE, POINTER, c_double, c_int, c_void_p
# Load the precompiled shared library
lib = ctypes.cdll.LoadLibrary('./libRION/libpso.so')

# Define the arguments and return type for the optimisation function
lib.optimize.argtypes = [c_void_p, c_void_p]
lib.optimize.restype = None

# call the function with the required parameters
lib.optimize(self._params, self._result)
```

On the C side, the function `optimize` corresponds to the function binding with the same name in Python.
```c
// Main optimization function with OpenMP parallelization
void optimize(PSOParams *params, PSOResult *result) {
    int num_particles = params->num_particles;
    int dim = params->dim;
    
    for (int iter = 0; iter < params->max_iter; iter++) {
        // Parallel particle update loop
        #pragma omp parallel
        {
            // Thread-local storage for potential global best updates
            double local_best_score = INFINITY;
            double *local_best_pos = (double*)malloc(dim * sizeof(double));
            
            #pragma omp for
            for (int p = 0; p < num_particles; p++) {
                double *pos = &params->positions[p * dim];
                double *vel = &params->velocities[p * dim];
                double *pbest = &params->personal_best_pos[p * dim];
                
                // Generate random values (thread-safe with OpenMP >= 3.0)
                unsigned int seed = omp_get_thread_num() + iter * 1000 + p;
                double r1 = (double)rand_r(&seed) / RAND_MAX;
                double r2 = (double)rand_r(&seed) / RAND_MAX;
                
                // Update velocity and position with bounds
                for (int d = 0; d < dim; d++) {
                    double cognitive = params->cognitive_coeff * r1 * (pbest[d] - pos[d]);
                    double social = params->social_coeff * r2 * (params->global_best_pos[d] - pos[d]);
                    vel[d] = params->inertia_weight * vel[d] + cognitive + social;
                    
                    // Velocity clamping
                    if (vel[d] > params->vmax[d]) vel[d] = params->vmax[d];
                    if (vel[d] < -params->vmax[d]) vel[d] = -params->vmax[d];
                    
                    pos[d] += vel[d];
                    
                    // Position clamping
                    if (pos[d] > params->upper_bounds[d]) pos[d] = params->upper_bounds[d];
                    if (pos[d] < params->lower_bounds[d]) pos[d] = params->lower_bounds[d];
                }
                
                // Evaluate new position
                double new_score = params->obj_func(pos, dim);
                
                // Update personal best
                if (new_score < params->personal_best_scores[p]) {
                    params->personal_best_scores[p] = new_score;
                    memcpy(pbest, pos, dim * sizeof(double));
                    
                    // Track local best for global update
                    if (new_score < local_best_score) {
                        local_best_score = new_score;
                        memcpy(local_best_pos, pos, dim * sizeof(double));
                    }
                }
            }
            
            // Critical section to update global best
            #pragma omp critical
            {
                if (local_best_score < params->global_best_score) {
                    params->global_best_score = local_best_score;
                    memcpy(params->global_best_pos, local_best_pos, dim * sizeof(double));
                }
            }
            
            free(local_best_pos);
        }
    }
    
    // Copy results
    memcpy(result->best_position, params->global_best_pos, dim * sizeof(double));
    result->best_score = params->global_best_score;
    memcpy(result->final_positions, params->positions, num_particles * dim * sizeof(double));
}
```
Thus parallel optimisation is acheived by using directive programming with `OpenMP` and interfacing with Python using the `ctypes` library.

## Building the project
To build this project, you will require `Python` to be installed on your system along with the tools, `Make` and `GCC`:

1. Create a virtual environment and activate it:
```sh
python -m venv .venv
source .venv/bin/activate
```

2. Navigate to the directory where you have cloned this repository and install the libraries from the `requirements.txt` file
```
pip install -r requirements.txt
```

3. Build the project using the Makefile provided:
```
make
```

## Output
After building the project, a basic summary of the optimisation is printed to the standard output along with the optimised population being stored to the directory `results/optimized_population.csv`

```txt
(.venv) $ make            
Checking dependencies for libRION/libPSO_parallel.py
python main.py
======================================================================
RION-Opt: Multi-Objective Optimization
======================================================================

Loading dataset: ./data/department_research_data.csv
✓ Dropped 'college_id' column
✓ Dataset loaded: 521 rows × 14 columns
  Columns: ['faculty_phd', 'faculty_other', 'frac_research_phd', 'frac_research_other', 'research_budget', 'centres_of_excellence_count', 'teaching_load_per_faculty', 'diversity_index', 'industry_collaborations', 'student_count', 'num_courses', 'papers_t1', 'citation_percentile_avg', 'overall_rank']

First row sample:
   faculty_phd  faculty_other  frac_research_phd  frac_research_other  research_budget  centres_of_excellence_count  teaching_load_per_faculty  diversity_index  industry_collaborations  student_count  num_courses  papers_t1  citation_percentile_avg  overall_rank
0          112             10           0.333588             0.330874     5.615820e+08                            2                   8.304167         0.992543                       12           1852           87       1123                 0.735653            58

✓ Defined 3 objective functions:
  1. Research Productivity (papers/FTE)
  2. Research Quality (impact/FTE)
  3. Cost Efficiency (budget/impact)

✓ Defined 14 constraints matching dataset columns

======================================================================
Initializing RION Optimizer
======================================================================
Configuration:
  Generations:     20
  Population Size: 50
  PSO Inertia:     0.5
  PSO Cognitive:   1.5
  PSO Social:      1.5
  PSO Iterations:  10

======================================================================
Running Optimization...
======================================================================


======================================================================
Optimization Complete!
======================================================================

✓ Results saved to: ./results/optimized_population.csv
  Final population size: 50

First few optimized solutions:
   faculty_phd  faculty_other  frac_research_phd  frac_research_other  ...  num_courses    papers_t1  citation_percentile_avg  overall_rank
0           16             11             0.1000             0.354428  ...    73.158942  1639.000000                 1.000000     59.000000
1           16             11             0.1000             0.050000  ...   117.796616  1639.000000                 1.000000     70.232054
2           16             17             0.1000             0.050000  ...    73.158942     0.000000                 1.000000     68.766880
3           16             17             0.3963             0.050000  ...    73.158942     1.644396                 0.642289     65.000000
4           16             11             0.1000             0.174091  ...    73.158942  1639.000000                 0.754973     79.000000

[5 rows x 14 columns]

Objective values for top 5 solutions:
Solution   Productivity    Quality         Cost/Impact    
-------------------------------------------------------
1          298.070         298.070         2.02e+02       
2          762.326         762.326         4.99e+05       
3          0.000           0.000           1.00e+05       
4          0.229           0.147           7.74e+08       
5          466.287         352.034         2.26e+05       

======================================================================
```