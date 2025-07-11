import numpy as np
import cma
import logging


# Name: AdaptiveIPOPIslandCMAESWithRuggedness
# Description:  Combines IPOP-CMA-ES with an island model and adapts sigma based on landscape ruggedness.
# Code:
class AdaptiveIPOPIslandCMAESWithRuggedness:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float],
                 num_islands: int = 5, initial_population_size: int = 100, initial_sigma: float = 5.0,
                 max_restarts: int = 30, population_size_multiplier: float = 3,
                 ruggedness_adaptation_rate: float = 0.1, large_population_size_factor: int = 100):
        """
        Initializes the Adaptive IPOP Island CMA-ES optimizer with ruggedness adaptation.

        Args:
            budget (int): Maximum number of function evaluations.
            dim (int): Problem dimensionality.
            lower_bounds (list[float]): List of lower bounds for each dimension.
            upper_bounds (list[float]): List of upper bounds for each dimension.
            num_islands (int): Number of islands in the island model.
            initial_population_size (int): Initial population size for each island.
            initial_sigma (float): Initial standard deviation for CMA-ES.
            max_restarts (int): Maximum number of restarts allowed for IPOP on each island.
            population_size_multiplier (float): Factor to increase population size on restart.
            ruggedness_adaptation_rate (float): Rate at which sigma adapts based on ruggedness.
            large_population_size_factor (int): factor to control max population size
        """
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)
        self.num_islands = num_islands
        self.initial_population_size = initial_population_size
        self.initial_sigma = initial_sigma
        self.max_restarts = max_restarts
        self.population_size_multiplier = population_size_multiplier
        self.ruggedness_adaptation_rate = ruggedness_adaptation_rate
        self.large_population_size_factor = large_population_size_factor

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        self.islands = []
        for _ in range(self.num_islands):
            self.islands.append({
                'population_size': initial_population_size,
                'best_solution': None,
                'best_fitness': float('inf'),
                'cma_es': None,
                'x0': (self.lower_bounds + self.upper_bounds) / 2.0,
                'sigma': initial_sigma,
                'ruggedness': 1.0,
                'num_restarts': 0 #restarts counter for ipop per island
            })

    def estimate_ruggedness(self, objective_function, x, sigma, n_samples=10):
        """Estimates the ruggedness of the landscape around a point x."""
        neighbor_fitnesses = []
        for _ in range(n_samples):
            neighbor = x + np.random.normal(0, sigma, self.dim)
            neighbor = np.clip(neighbor, self.lower_bounds, self.upper_bounds)
            neighbor_fitnesses.append(objective_function(neighbor.reshape(1, -1))[0])
        return np.std(neighbor_fitnesses)

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8, optimum_value=None) -> tuple:
        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        while self.eval_count < self.budget:
            for i in range(self.num_islands):
                island = self.islands[i]

                if island['cma_es'] is None:  # Initialize or restart CMA-ES
                    island['x0'] = (self.lower_bounds + self.upper_bounds) / 2.0 # Reset mean on restart

                    options = {
                        'bounds': [list(self.lower_bounds), list(self.upper_bounds)],
                        'popsize': island['population_size'],
                        'maxfevals': self.budget - self.eval_count,
                        'verb_disp': 0,
                        'seed': np.random.randint(0, 1000000),
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
                        island['cma_es'] = cma.fmin(cma_objective_wrapper, island['x0'], island['sigma'], options)
                        result_solution = island['cma_es'][0]
                        result_fitness = island['cma_es'][1]

                        if result_fitness < island['best_fitness']:
                            island['best_fitness'] = result_fitness
                            island['best_solution'] = result_solution

                        if result_fitness < self.best_fitness_overall:
                            self.best_fitness_overall = result_fitness
                            self.best_solution_overall = result_solution

                    except Exception as e:
                        logging.error(f"CMA-ES run on island {i} failed due to: {e}", exc_info=True)

                #IPOP Restart Logic
                if island['cma_es'] is not None and island['num_restarts'] < self.max_restarts:
                    island['num_restarts'] +=1
                    # Estimate ruggedness around the best solution of the island
                    ruggedness = self.estimate_ruggedness(objective_function, island['best_solution'], island['sigma'])
                    island['ruggedness'] = ruggedness
                    island['sigma'] *= np.exp(self.ruggedness_adaptation_rate * (ruggedness - 1))  # Adapt sigma

                    island['population_size'] = int(island['population_size'] * self.population_size_multiplier)

                    if island['population_size'] > self.large_population_size_factor * self.dim:
                        island['population_size'] = self.large_population_size_factor * self.dim
                        logging.warning(f"Island {i}: Population size capped to {island['population_size']}")

                    island['cma_es'] = None #flag for restart

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall,
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info