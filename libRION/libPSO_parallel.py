import numpy as np
import ctypes
from ctypes import CFUNCTYPE, POINTER, c_double, c_int, c_void_p

# Load the shared library
lib = ctypes.cdll.LoadLibrary('./libRION/libpso.so')

# Define callback type for objective function
OBJECTIVE_FUNC = CFUNCTYPE(c_double, POINTER(c_double), c_int)

# Define C function signatures
lib.pso_create.argtypes = [c_int, c_int, c_int, c_double, c_double, c_double,
                           POINTER(c_double), POINTER(c_double),
                           POINTER(c_double), POINTER(c_double),
                           OBJECTIVE_FUNC]
lib.pso_create.restype = c_void_p

lib.pso_free.argtypes = [c_void_p]
lib.pso_free.restype = None

lib.result_create.argtypes = [c_int, c_int]
lib.result_create.restype = c_void_p

lib.result_free.argtypes = [c_void_p]
lib.result_free.restype = None

lib.optimize.argtypes = [c_void_p, c_void_p]
lib.optimize.restype = None

lib.get_results.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double),
                            POINTER(c_double), c_int, c_int]
lib.get_results.restype = None


class PSO:
    def __init__(self, obj_func, data, num_particles=30, max_iter=100, 
                 inertia_weight=0.5, cognitive_coeff=1.5, social_coeff=1.5,
                 lower_bounds=None, upper_bounds=None):
        """
        Initialize the PSO optimizer with C/OpenMP backend.
        
        Parameters:
        - obj_func: function to minimize (should take a vector and return a scalar)
        - data: numpy array or list of input vectors (initial population)
        - num_particles: number of particles
        - max_iter: number of iterations
        - inertia_weight: inertia weight for velocity update
        - cognitive_coeff: cognitive coefficient
        - social_coeff: social coefficient
        - lower_bounds: array of lower bounds for each dimension (optional)
        - upper_bounds: array of upper bounds for each dimension (optional)
        """
        
        self.obj_func = obj_func
        self.data = np.asarray(data, dtype=np.float64)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        
        self.dim = self.data.shape[1]
        
        # Set bounds - infer from data if not provided
        if lower_bounds is None:
            data_min = np.min(data, axis=0)
            data_range = np.max(data, axis=0) - data_min
            self.lower_bounds = data_min - 0.5 * np.abs(data_range)
        else:
            self.lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        
        if upper_bounds is None:
            data_max = np.max(data, axis=0)
            data_range = data_max - np.min(data, axis=0)
            self.upper_bounds = data_max + 0.5 * np.abs(data_range)
        else:
            self.upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
        
        # Prepare initial positions and velocities
        self.positions = np.copy(self.data).astype(np.float64)
        self.velocities = np.random.uniform(-1, 1, (num_particles, self.dim)).astype(np.float64)
        
        # Create C-callable callback function
        @OBJECTIVE_FUNC
        def c_objective(pos_ptr, dim):
            # Convert C pointer to numpy array (no copy)
            pos_array = np.ctypeslib.as_array(pos_ptr, shape=(dim,))
            return float(self.obj_func(pos_array))
        
        # Store callback to prevent garbage collection
        self._c_objective = c_objective
        
        # Create C pointers
        positions_ptr = self.positions.ctypes.data_as(POINTER(c_double))
        velocities_ptr = self.velocities.ctypes.data_as(POINTER(c_double))
        lower_bounds_ptr = self.lower_bounds.ctypes.data_as(POINTER(c_double))
        upper_bounds_ptr = self.upper_bounds.ctypes.data_as(POINTER(c_double))
        
        # Create PSO parameters structure in C
        self._params = lib.pso_create(
            num_particles, self.dim, max_iter,
            inertia_weight, cognitive_coeff, social_coeff,
            positions_ptr, velocities_ptr,
            lower_bounds_ptr, upper_bounds_ptr,
            self._c_objective
        )
        
        # Create result structure
        self._result = lib.result_create(num_particles, self.dim)
    
    def optimize(self):
        """
        Run PSO optimization using C/OpenMP backend.
        
        Returns:
        - global_best_position: Best position found
        - global_best_score: Best score found
        """
        # Run optimization
        lib.optimize(self._params, self._result)
        
        # Prepare arrays to receive results
        best_position = np.zeros(self.dim, dtype=np.float64)
        final_positions = np.zeros((self.num_particles, self.dim), dtype=np.float64)
        best_score = c_double()
        
        # Extract results using helper function
        lib.get_results(
            self._result,
            best_position.ctypes.data_as(POINTER(c_double)),
            ctypes.byref(best_score),
            final_positions.ctypes.data_as(POINTER(c_double)),
            self.num_particles,
            self.dim
        )
        
        # Update instance variables to match original API
        self.positions = final_positions
        self.global_best_position = best_position
        self.global_best_score = best_score.value
        
        return best_position, best_score.value
    
    def __del__(self):
        """Cleanup C resources"""
        if hasattr(self, '_params') and self._params:
            lib.pso_free(self._params)
        if hasattr(self, '_result') and self._result:
            lib.result_free(self._result)
