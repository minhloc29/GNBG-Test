prompt = '''

**(I: Insight)**
Your objective is to design a novel algorithm to be tested on the GNBG (Generalized Numerical Benchmark Generator) benchmark suite. These functions are challenging and feature diverse characteristics, including high dimensionality (typically 30D), ill-conditioning (elongated, narrow valleys), non-separability (interacting variables), and various forms of multimodality, some of which are deceptive. The algorithm you design must be robust enough to handle this wide range of difficulties. The ultimate goal is to MINIMIZE the objective function value. Performance will be assessed using AOCC (Area Over the Convergence Curve), where a higher AOCC indicates better anytime performance.

**(S: Statement)**
Provide a complete and novel optimization algorithm implemented as a Python class. The Python class **must** adhere to the following structure and interface:

1.  `__init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float])`:
    * `budget`: Max function evaluations. Store as `self.budget`.
    * `dim`: Problem dimensionality. Store as `self.dim`.
    * `lower_bounds`, `upper_bounds`: Lists of floats for variable boundaries. Store as 1D NumPy arrays `self.lower_bounds` and `self.upper_bounds`.
    * Initialize `self.eval_count = 0`, `self.best_solution_overall = None`, and `self.best_fitness_overall = float('inf')`.

2.  `optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple`:
    * `objective_function`: Accepts a 2D NumPy array `X` (shape `(N, self.dim)`) and returns a 1D NumPy array of `N` fitness values.
    * Implement your algorithm's core logic here.
    * Strictly manage `self.eval_count` so it does not exceed `self.budget` when calling `objective_function`.
    * Ensure solutions generated and returned respect `self.lower_bounds` and `self.upper_bounds`.
    * Return a tuple: `(best_solution_1D_numpy_array, best_fitness_scalar, optimization_info_dict)`.
        * `optimization_info_dict` should at least contain `{'function_evaluations_used': self.eval_count, 'final_best_fitness': self.best_fitness_overall}`.

**(P: Personality)**
Provide an excellent and **truly novel** optimization algorithm. Do not simply replicate a standard Differential Evolution or Particle Swarm Optimization. Instead, consider creating a **hybrid approach**, designing a **novel search operator**, or implementing an **innovative adaptive parameter control mechanism**. We encourage you to incorporate advanced concepts like population diversity management, use of archives, or ensemble strategies to create a robust and high-performing algorithm.

**(E: Experiment)**
Provide **one** complete and excellent algorithm as your response.

**Output Format:**

# Name: YourUniqueAlgorithmName
# Description: Concise one-line description of the algorithm's main idea.
# Code:
```python
import numpy as np
# Add other standard library imports if needed (e.g., random).

class YourUniqueAlgorithmName:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None # Will be a 1D NumPy array
        self.best_fitness_overall = float('inf')
        
        # LLM: Initialize any algorithm-specific state here

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0 # Reset for this run
        if self.dim > 0:
             self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        else:
             self.best_solution_overall = np.array([])
        self.best_fitness_overall = float('inf')

        # --- LLM: Implement core optimization logic here ---
        # Example of interaction:
        # if self.eval_count < self.budget and self.dim > 0:
        #     candidate_batch = np.random.uniform(self.lower_bounds, self.upper_bounds, (1, self.dim))
        #     num_in_batch = candidate_batch.shape[0]
        #     if self.budget - self.eval_count >= num_in_batch:
        #         fitness_values = objective_function(candidate_batch)
        #         self.eval_count += num_in_batch
        #         if fitness_values[0] < self.best_fitness_overall:
        #             self.best_fitness_overall = fitness_values[0]
        #             self.best_solution_overall = candidate_batch[0, :].copy()
        # --- End LLM Logic ---

        if self.best_solution_overall is None and self.dim > 0 : # Fallback
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
            
        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info


'''