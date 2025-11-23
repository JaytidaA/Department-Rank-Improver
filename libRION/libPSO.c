#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// Callback function type for objective function
typedef double (*ObjectiveFunc)(double*, int);

typedef struct {
    double *positions;           // num_particles x dim
    double *velocities;          // num_particles x dim
    double *personal_best_pos;   // num_particles x dim
    double *personal_best_scores; // num_particles
    double *global_best_pos;     // dim
    double global_best_score;
    
    double *lower_bounds;        // dim
    double *upper_bounds;        // dim
    double *vmax;                // dim
    
    int num_particles;
    int dim;
    int max_iter;
    double inertia_weight;
    double cognitive_coeff;
    double social_coeff;
    
    ObjectiveFunc obj_func;
} PSOParams;

typedef struct {
    double *best_position;
    double best_score;
    double *final_positions;  // num_particles x dim
} PSOResult;

// Allocate PSO parameters
PSOParams* pso_create(int num_particles, int dim, int max_iter,
                      double inertia, double cognitive, double social,
                      double *initial_positions, double *initial_velocities,
                      double *lower_bounds, double *upper_bounds,
                      ObjectiveFunc obj_func) {
    PSOParams *params = (PSOParams*)malloc(sizeof(PSOParams));
    
    params->num_particles = num_particles;
    params->dim = dim;
    params->max_iter = max_iter;
    params->inertia_weight = inertia;
    params->cognitive_coeff = cognitive;
    params->social_coeff = social;
    params->obj_func = obj_func;
    
    int total_size = num_particles * dim;
    
    params->positions = (double*)malloc(total_size * sizeof(double));
    params->velocities = (double*)malloc(total_size * sizeof(double));
    params->personal_best_pos = (double*)malloc(total_size * sizeof(double));
    params->personal_best_scores = (double*)malloc(num_particles * sizeof(double));
    params->global_best_pos = (double*)malloc(dim * sizeof(double));
    
    params->lower_bounds = (double*)malloc(dim * sizeof(double));
    params->upper_bounds = (double*)malloc(dim * sizeof(double));
    params->vmax = (double*)malloc(dim * sizeof(double));
    
    memcpy(params->positions, initial_positions, total_size * sizeof(double));
    memcpy(params->velocities, initial_velocities, total_size * sizeof(double));
    memcpy(params->personal_best_pos, initial_positions, total_size * sizeof(double));
    memcpy(params->lower_bounds, lower_bounds, dim * sizeof(double));
    memcpy(params->upper_bounds, upper_bounds, dim * sizeof(double));
    
    // Calculate vmax as 20% of search space range
    for (int d = 0; d < dim; d++) {
        params->vmax[d] = 0.2 * (upper_bounds[d] - lower_bounds[d]);
    }
    
    // Clamp initial velocities
    for (int i = 0; i < total_size; i++) {
        int d = i % dim;
        if (params->velocities[i] > params->vmax[d]) {
            params->velocities[i] = params->vmax[d];
        }
        if (params->velocities[i] < -params->vmax[d]) {
            params->velocities[i] = -params->vmax[d];
        }
    }
    
    // Initialize personal best scores and find global best
    params->global_best_score = INFINITY;
    for (int p = 0; p < num_particles; p++) {
        double score = obj_func(&params->positions[p * dim], dim);
        params->personal_best_scores[p] = score;
        
        if (score < params->global_best_score) {
            params->global_best_score = score;
            memcpy(params->global_best_pos, &params->positions[p * dim], dim * sizeof(double));
        }
    }
    
    return params;
}

// Free PSO parameters
void pso_free(PSOParams *params) {
    if (params) {
        free(params->positions);
        free(params->velocities);
        free(params->personal_best_pos);
        free(params->personal_best_scores);
        free(params->global_best_pos);
        free(params->lower_bounds);
        free(params->upper_bounds);
        free(params->vmax);
        free(params);
    }
}

// Allocate result structure
PSOResult* result_create(int num_particles, int dim) {
    PSOResult *result = (PSOResult*)malloc(sizeof(PSOResult));
    result->best_position = (double*)malloc(dim * sizeof(double));
    result->final_positions = (double*)malloc(num_particles * dim * sizeof(double));
    return result;
}

// Free result structure
void result_free(PSOResult *result) {
    if (result) {
        free(result->best_position);
        free(result->final_positions);
        free(result);
    }
}

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
        
        // printf("Iteration %d/%d, Global Best Score: %f\n", 
        //        iter + 1, params->max_iter, params->global_best_score);
    }
    
    // Copy results
    memcpy(result->best_position, params->global_best_pos, dim * sizeof(double));
    result->best_score = params->global_best_score;
    memcpy(result->final_positions, params->positions, num_particles * dim * sizeof(double));
}

// Helper function to extract results
void get_results(PSOResult *result, double *best_pos_out, double *best_score_out, 
                 double *final_pos_out, int num_particles, int dim) {
    memcpy(best_pos_out, result->best_position, dim * sizeof(double));
    *best_score_out = result->best_score;
    memcpy(final_pos_out, result->final_positions, num_particles * dim * sizeof(double));
}
