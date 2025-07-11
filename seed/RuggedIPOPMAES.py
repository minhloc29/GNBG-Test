import numpy as np
import random
import cma
import logging

# Name: RuggedIPOPCMAES
# Description: Combines IPOP restarts with landscape ruggedness-based sigma adaptation for multi-basin exploration.

class RuggedIPOPCMAES:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float],
                 initial_population_size = None, initial_sigma: float = 0.2,
                 max_restarts: int = 10, population_size_multiplier: float = 2.0,
                 large_population_size_factor: int = 20):
        """
        Initializes the RuggedIPOPCMAES optimizer.

        Args:
            budget (int): Maximum number of function evaluations.
            dim (int): Problem dimensionality.
            lower_bounds (list[float]): List of lower bounds for each dimension.
            upper_bounds (list[float]): List of upper bounds for each dimension.
            initial_population_size (Optional[int]): Initial population size. If None, CMA-ES default is used.
            initial_sigma (float): Initial standard deviation for search.
            max_restarts (int): Maximum number of restarts allowed.
            population_size_multiplier (float): Factor by which population size increases on restart.
            large_population_size_factor (int): Factor to determine the large population size threshold.
        """
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)
        self.initial_population_size = initial_population_size if initial_population_size else (4 + int(np.floor(3 * np.log(dim))))
        self.initial_sigma = initial_sigma
        self.max_restarts = max_restarts
        self.population_size_multiplier = population_size_multiplier
        self.large_population_size_factor = large_population_size_factor

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

    def calculate_ruggedness(self, objective_function: callable, x: np.ndarray, delta: float = 0.01):
        """Estimates landscape ruggedness around a point."""
        fitness_center = objective_function(x.reshape(1, -1))[0]
        neighbors = []
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += delta
            x_minus[i] -= delta
            x_plus = np.clip(x_plus, self.lower_bounds[i], self.upper_bounds[i])
            x_minus = np.clip(x_minus, self.lower_bounds[i], self.upper_bounds[i])
            neighbors.append(x_plus)
            neighbors.append(x_minus)

        neighbor_fitnesses = objective_function(np.array(neighbors))
        ruggedness = np.std(neighbor_fitnesses)
        return ruggedness

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8, optimum_value=None,
                 cmaes_verbosity: int = 0, cmaes_seed_upper_bound: int = 1000000):
        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        current_population_size = self.initial_population_size
        num_restarts = 0

        while self.eval_count < self.budget and num_restarts <= self.max_restarts:
            logging.info(f"Rugged-IPOP-CMA-ES: Starting new run (restart {num_restarts + 1}) "
                         f"with population size {current_population_size} at FE={self.eval_count}")

            x0 = (self.lower_bounds + self.upper_bounds) / 2.0

            # Adapt sigma based on landscape ruggedness
            ruggedness = self.calculate_ruggedness(objective_function, x0)
            adapted_sigma = self.initial_sigma * (1 + ruggedness)  # Increase sigma in rugged landscapes

            options = {
                'bounds': [list(self.lower_bounds), list(self.upper_bounds)],
                'popsize': current_population_size,
                'maxfevals': self.budget - self.eval_count, # Remaining budget for this run
                'verb_disp': cmaes_verbosity,
                'seed': np.random.randint(0, cmaes_seed_upper_bound),
            }

            def cma_objective_wrapper(x):
                self.eval_count += 1
                if self.eval_count > self.budget:
                    return float('inf')

                if optimum_value is not None and abs(self.best_fitness_overall - optimum_value) <= acceptance_threshold:
                    return float('inf')

                fitness_val = objective_function(x.reshape(1, -1))[0]

                if fitness_val < self.best_fitness_overall:
                    self.best_fitness_overall = fitness_val
                    self.best_solution_overall = x.copy()
                return fitness_val

            try:
                result = cma.fmin(cma_objective_wrapper, x0, adapted_sigma, options)
                result_solution = result[0]
                result_fitness = result[1]
                evaluations_used_by_cma = result[2]
                iterations_cma = result[3]

                if result_fitness < self.best_fitness_overall:
                    self.best_fitness_overall = result_fitness
                    self.best_solution_overall = result_solution

                if self.eval_count >= self.budget or (optimum_value is not None and abs(self.best_fitness_overall - optimum_value) <= acceptance_threshold):
                    break

                logging.info(f"CMA-ES run completed. Best fitness: {result_fitness:.6e}, FE used: {evaluations_used_by_cma}")

            except Exception as e:
                logging.error(f"CMA-ES run failed due to: {e}", exc_info=True)
                if self.eval_count >= self.budget:
                    break

            num_restarts += 1
            current_population_size = int(current_population_size * self.population_size_multiplier)

            if current_population_size > self.large_population_size_factor * self.dim:
                logging.warning(f"IPOP-CMA-ES: Population size grew very large ({current_population_size}). Capping for next restart.")
                current_population_size = self.large_population_size_factor * self.dim

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall,
            'num_restarts': num_restarts
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info