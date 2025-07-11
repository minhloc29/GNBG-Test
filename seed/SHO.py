import numpy as np
import logging

class IPOPSeahorse:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float],
                 initial_population_size=100, max_restarts: int = 50,
                 population_size_multiplier: float = 3.0):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)
        self.initial_population_size = initial_population_size
        self.max_restarts = max_restarts
        self.population_size_multiplier = population_size_multiplier

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8, optimum_value=None):
        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        current_population_size = self.initial_population_size
        num_restarts = 0

        while self.eval_count < self.budget and num_restarts <= self.max_restarts:
            logging.info(f"IPOP-SHO: Restart {num_restarts+1} with population size {current_population_size}, FE={self.eval_count}")

            # Initialize population
            population = np.random.uniform(self.lower_bounds, self.upper_bounds, (current_population_size, self.dim))
            fitness = np.array([objective_function(ind.reshape(1, -1))[0] for ind in population])
            self.eval_count += current_population_size

            best_idx = np.argmin(fitness)
            best_solution = population[best_idx].copy()
            best_fitness = fitness[best_idx]

            if best_fitness < self.best_fitness_overall:
                self.best_fitness_overall = best_fitness
                self.best_solution_overall = best_solution.copy()

            if optimum_value is not None and abs(self.best_fitness_overall - optimum_value) <= acceptance_threshold:
                break

            t = 0
            max_iter = 1000  # or until budget is exhausted

            while self.eval_count < self.budget and t < max_iter:
                a = 2 - 2 * t / max_iter  # Like WOA or SHO's decay factor
                for i in range(current_population_size):
                    r = np.random.rand(self.dim)
                    spiral = np.sin(r * np.pi * 2)

                    if np.random.rand() < 0.5:
                        new_position = population[i] + a * spiral * (best_solution - population[i])
                    else:
                        j = np.random.randint(current_population_size)
                        if j != i:
                            new_position = population[i] + a * np.random.rand() * (population[j] - population[i])
                        else:
                            new_position = population[i]

                    # Clip to bounds
                    new_position = np.clip(new_position, self.lower_bounds, self.upper_bounds)

                    # Evaluate
                    new_fitness = objective_function(new_position.reshape(1, -1))[0]
                    self.eval_count += 1

                    # Update if better
                    if new_fitness < fitness[i]:
                        population[i] = new_position
                        fitness[i] = new_fitness

                        if new_fitness < self.best_fitness_overall:
                            self.best_fitness_overall = new_fitness
                            self.best_solution_overall = new_position.copy()

                    if self.eval_count >= self.budget or (optimum_value is not None and abs(self.best_fitness_overall - optimum_value) <= acceptance_threshold):
                        break
                t += 1

            logging.info(f"SHO run done. Best fitness: {self.best_fitness_overall:.6e}, FE used: {self.eval_count}")

            num_restarts += 1
            current_population_size = int(current_population_size * self.population_size_multiplier)
            if current_population_size > 100 * self.dim:
                logging.warning(f"IPOP-SHO: Population too large ({current_population_size}), capping.")
                current_population_size = 100 * self.dim

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall,
            'num_restarts': num_restarts
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info