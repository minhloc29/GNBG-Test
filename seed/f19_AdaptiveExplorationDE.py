import numpy as np
import random

# Name: AdaptiveExplorationDifferentialEvolution
# Description: A differential evolution algorithm that dynamically adjusts its exploration rate based on landscape characteristics.

class AdaptiveExplorationDifferentialEvolution:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 50  # Adjust population size
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        self.mutation_factor = 0.5  # Mutation factor
        self.crossover_rate = 0.7  # Crossover rate
        self.exploration_prob = 0.1  # Probability of exploration

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0  # Reset for this run
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim)) # Re-initialize population
        if self.dim > 0:
             self.best_solution_overall = self.population[0].copy()  # Initialize with a random member
        else:
             self.best_solution_overall = np.array([])
        
        fitness = objective_function(self.population)
        self.eval_count += self.population_size
        self.best_fitness_overall = np.min(fitness)
        self.best_solution_overall = self.population[np.argmin(fitness)].copy()


        while self.eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                
                # Adaptive Exploration: Introduce random jumps with probability exploration_prob
                if random.random() < self.exploration_prob:
                    mutant = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
                else:
                    mutant = a + self.mutation_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds)  # Keep within bounds


                # Crossover
                trial = np.copy(self.population[i])
                for j in range(self.dim):
                    if random.random() < self.crossover_rate:
                        trial[j] = mutant[j]
                    
                trial = np.clip(trial, self.lower_bounds, self.upper_bounds)

                # Selection
                trial_fitness = objective_function(trial.reshape(1, -1))[0] # Evaluate trial vector
                self.eval_count += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    self.population[i] = trial.copy()

                    if trial_fitness < self.best_fitness_overall:
                        self.best_fitness_overall = trial_fitness
                        self.best_solution_overall = trial.copy()
                        
                # Dynamically adjust exploration probability (example: decrease as progress is made)
                self.exploration_prob = max(0.01, 0.1 - (self.eval_count / self.budget) * 0.09)


        if self.best_solution_overall is None and self.dim > 0: # Fallback
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
            
        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info