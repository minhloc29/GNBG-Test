import numpy as np
import random

class HybridDEArchiveAdaptive:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        self.population_size = int(5 + np.ceil(np.log(self.dim))) # Population scaling with dimension
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))

        self.archive_size = self.population_size * 2  # Archive is larger than the population
        self.archive = np.zeros((self.archive_size, self.dim))
        self.archive_fitness = np.full(self.archive_size, float('inf'))
        self.archive_idx = 0

        self.mutation_factor = 0.5  # Initial mutation factor
        self.crossover_rate = 0.7 # Initial crossover rate
        self.mutation_factor_history = [self.mutation_factor]
        self.crossover_rate_history = [self.crossover_rate]
        self.min_archive_fitness = float('inf')  #ADDED: Minimum acceptable fitness for archive entry

    def ensure_bounds(self, vec):
        return np.clip(vec, self.lower_bounds, self.upper_bounds)

    def update_archive(self, individual, fitness):
        if fitness < np.max(self.archive_fitness) and fitness < self.min_archive_fitness:
            worst_index = np.argmax(self.archive_fitness)
            self.archive[worst_index] = individual
            self.archive_fitness[worst_index] = fitness
            self.min_archive_fitness = np.min(self.archive_fitness)

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        if self.dim > 0:
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        else:
            self.best_solution_overall = np.array([])
        self.best_fitness_overall = float('inf')

        # Initialize population and fitness
        fitness_values = objective_function(self.population)
        self.eval_count += self.population_size
        self.fitness = fitness_values.copy()

        for i in range(self.population_size):
            if self.fitness[i] < self.best_fitness_overall:
                self.best_fitness_overall = self.fitness[i]
                self.best_solution_overall = self.population[i].copy()

        # Initialize Archive with initial population
        self.archive[:self.population_size] = self.population.copy()
        self.archive_fitness[:self.population_size] = self.fitness.copy()
        self.min_archive_fitness = np.max(self.archive_fitness) #initialize min_archive_fitness
        self.archive_idx = self.population_size

        while self.eval_count < self.budget and self.dim > 0:
            for i in range(self.population_size):
                # Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                if len(indices) >= 3: #Ensure we have at least 3 distinct indices from pop
                    a, b, c = random.sample(indices, 3)
                else:
                    a, b, c = random.choices(indices, k = 3) #Allows replacement to proceed if len(indices) < 3

                #Potentially use an archive member
                if random.random() < 0.1: #10% chance of pulling a member from the archive
                    arc_idx = random.randint(0, self.archive_size - 1)
                    mutant = self.population[a] + self.mutation_factor * (self.archive[arc_idx] - self.population[b]) #Simplified
                else:
                    mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c]) #+ self.mutation_factor * (self.best_solution_overall - self.population[i]) #Best solution guided mutation Removed due to line limit

                mutant = self.ensure_bounds(mutant)
                
                # Crossover
                trial_vector = np.zeros(self.dim)
                for j in range(self.dim):
                    if random.random() < self.crossover_rate or j == random.randint(0, self.dim - 1):
                        trial_vector[j] = mutant[j]
                    else:
                        trial_vector[j] = self.population[i][j]

                trial_vector = self.ensure_bounds(trial_vector)

                # Selection
                trial_fitness = objective_function(trial_vector.reshape(1, -1))[0]
                self.eval_count += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector.copy()
                    self.fitness[i] = trial_fitness

                    # Update best solution
                    if trial_fitness < self.best_fitness_overall:
                        self.best_fitness_overall = trial_fitness
                        self.best_solution_overall = trial_vector.copy()

                    #Update Archive
                    self.update_archive(trial_vector, trial_fitness)

                #else: #Removed because it exceeds change limit
                    ##Decrease mutation and crossover if no improvement
                    #self.mutation_factor = max(0.1, self.mutation_factor * 0.9)
                    #self.crossover_rate = max(0.1, self.crossover_rate * 0.9)

                self.mutation_factor_history.append(self.mutation_factor)
                self.crossover_rate_history.append(self.crossover_rate)
            worst_index = np.argmax(self.fitness)
            self.mutation_factor = max(0.1, min(1.0, self.mutation_factor * (1 + 0.1 * np.std(self.fitness) / (self.fitness[worst_index] + 1e-8))))

        if self.best_solution_overall is None and self.dim > 0:
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall,
            'mutation_factor_history': self.mutation_factor_history,
            'crossover_rate_history': self.crossover_rate_history
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info