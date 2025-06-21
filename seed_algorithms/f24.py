import numpy as np
import random

class AdaptiveGaussianMutationDE:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 100  # Adjust as needed
        self.population = None
        self.fitness_values = None
        self.mutation_scale = 0.8 # Initial mutation scale
        self.mutation_scale_decay = 0.99 #decay factor for the mutation scale

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.population = self._initialize_population()
        self.fitness_values = self._evaluate_population(objective_function)

        self.best_solution_overall, self.best_fitness_overall = self._find_best(self.population,self.fitness_values)

        while self.eval_count < self.budget:
            new_population = []
            new_fitness_values = []

            for i in range(self.population_size):
                # Differential Mutation
                a, b, c = self._select_different(i)
                mutant = self.population[a] + self.mutation_scale * (self.population[b] - self.population[c])

                #Adaptive Gaussian perturbation to escape local optima
                mutant += np.random.normal(0, self.mutation_scale/2, self.dim)  

                #Clipping
                mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds)

                #Crossover
                trial = np.where(np.random.rand(self.dim) < 0.5, mutant, self.population[i])

                #Selection
                trial_fitness = objective_function(trial.reshape(1, -1))[0]
                self.eval_count += 1
                if trial_fitness < self.fitness_values[i]:
                    new_population.append(trial)
                    new_fitness_values.append(trial_fitness)
                else:
                    new_population.append(self.population[i])
                    new_fitness_values.append(self.fitness_values[i])
                
                best_solution,best_fitness = self._find_best(np.array(new_population), np.array(new_fitness_values))
                if best_fitness < self.best_fitness_overall:
                    self.best_solution_overall = best_solution
                    self.best_fitness_overall = best_fitness


            self.population = np.array(new_population)
            self.fitness_values = np.array(new_fitness_values)
            self.mutation_scale *= self.mutation_scale_decay #Decay mutation scale

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info

    def _initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, size=(self.population_size, self.dim))

    def _evaluate_population(self, objective_function):
        population_reshaped = self.population.reshape(-1, self.dim)
        fitness = objective_function(population_reshaped)
        self.eval_count += self.population_size
        return fitness

    def _select_different(self, index):
        a, b, c = random.sample(range(self.population_size), 3)
        while a == index or b == index or c == index or a == b or a == c or b == c:
            a, b, c = random.sample(range(self.population_size), 3)
        return a, b, c

    def _find_best(self,population,fitness_values):
        best_index = np.argmin(fitness_values)
        return population[best_index], fitness_values[best_index]