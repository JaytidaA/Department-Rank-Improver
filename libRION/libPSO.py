import numpy as np

class PSO:
    def __init__ (self, obj_func, data, num_particles=30, max_iter=100, inertia_weight=0.5, cognitive_coeff=1.5, social_coeff=1.5):
        """
        Initialize the PSO optimizer.

        Parameters:
        - obj_function: function to minimize (should take a vector and return a scalar)
        - data: numpy array or list of input vectors (initial population)
        - num_particles: number of particles
        - max_iter: number of iterations
        - inertia_weight: inertia weight for velocity update
        - cognitive_coeff: cognitive coefficient
        - social_coeff: social coefficient
        """

        self.obj_func = obj_func
        self.data = data
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff

        self.dim = self.data.shape[1]
        
        # Cast to float64 to avoid casting errors
        self.positions = np.copy(self.data).astype(np.float64)
        self.velocities = np.random.uniform(-1, 1, (num_particles, self.dim)).astype(np.float64)

        # Initialize personal best positions and global best position
        self.personal_best_positions = np.copy(self.positions).astype(np.float64)
        self.personal_best_scores = np.array([self.obj_func(x) for x in self.positions])
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_scores)].astype(np.float64)
        self.global_best_score = np.min(self.personal_best_scores)

    def optimize(self):
        for i in range(self.max_iter):
            for p in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive = self.cognitive_coeff * r1 * (self.personal_best_positions[p] - self.positions[p])
                social = self.social_coeff * r2 * (self.global_best_position - self.positions[p])
                self.velocities[p] = (self.inertia_weight * self.velocities[p] + cognitive + social)

                self.positions[p] += self.velocities[p]

                new_score = self.obj_func(self.positions[p])
                if new_score < self.personal_best_scores[p]:
                    self.personal_best_scores[p] = new_score
                    self.personal_best_positions[p] = np.copy(self.positions[p])

                    if new_score < self.global_best_score:
                        self.global_best_score = new_score
                        self.global_best_position = np.copy(self.positions[p])
            
            print(f"Iteration {i+1}/{self.max_iter}, Global Best Score: {self.global_best_score}")

        return self.global_best_position, self.global_best_score
