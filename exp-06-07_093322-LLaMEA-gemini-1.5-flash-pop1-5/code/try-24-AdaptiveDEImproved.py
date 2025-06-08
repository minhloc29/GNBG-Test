import numpy as np
import random

class AdaptiveDEImproved:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 10 * self.dim  # Adaptive population size
        self.F = 0.8  # Scaling factor for mutation
        self.CR = 0.9  # Crossover rate
        self.F_adapt = 0.5 # Initial adaptive mutation factor
        self.density_threshold = 0.8

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        self.best_fitness_overall = objective_function(np.array([self.best_solution_overall]))[0]
        self.eval_count += 1

        population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        fitness_values = objective_function(population)
        self.eval_count += self.population_size

        for i in range(self.population_size):
            if fitness_values[i] < self.best_fitness_overall:
                self.best_fitness_overall = fitness_values[i]
                self.best_solution_overall = population[i, :].copy()

        while self.eval_count < self.budget:
            new_population = np.zeros_like(population)
            for i in range(self.population_size):
                # Adaptive Mutation Strategy
                a, b, c = random.sample(range(self.population_size), 3)
                while a == i or b == i or c == i:
                    a, b, c = random.sample(range(self.population_size), 3)

                mutant = population[a] + self.F_adapt * (population[b] - population[c])

                # Local Search Enhancement:  Adjust mutation based on local fitness and density
                distances = np.linalg.norm(population - population[i], axis=1)
                density = np.sum(distances < np.mean(distances)) / self.population_size

                if density > self.density_threshold:
                    self.F_adapt = min(1.0, self.F_adapt + 0.1) #Increase Mutation
                else:
                    self.F_adapt = max(0.1, self.F_adapt - 0.05) #Decrease Mutation


                #Boundary Handling
                mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds)

                #Crossover
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])

                #Selection
                trial_fitness = objective_function(np.array([trial]))[0]
                self.eval_count += 1
                if trial_fitness < fitness_values[i]:
                    new_population[i] = trial
                    fitness_values[i] = trial_fitness
                    if trial_fitness < self.best_fitness_overall:
                        self.best_fitness_overall = trial_fitness
                        self.best_solution_overall = trial.copy()

            population = new_population


        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info