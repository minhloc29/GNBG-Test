import numpy as np
from scipy.spatial.distance import pdist, squareform

class AdaptiveDEwithLocalSearch:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 10 * self.dim #Adaptive population size
        self.F = 0.8  # Differential Evolution scaling factor
        self.CR = 0.9 # Crossover rate

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        fitness_values = objective_function(self.population)
        self.eval_count += self.population_size
        
        self.best_solution_overall = self.population[np.argmin(fitness_values)]
        self.best_fitness_overall = np.min(fitness_values)

        for _ in range(int(self.budget / self.population_size)):
          
            #Adaptive Mutation Strategy
            diversity = np.mean(pdist(self.population))
            if diversity < 0.1: # Adjust this threshold based on problem's complexity
                self.F = 0.2 # Reduce exploration in high diversity
            else:
                self.F = 0.8 # Increase exploration in low diversity
                
            offspring = self.mutate(self.population, self.F, self.CR)
            offspring = np.clip(offspring, self.lower_bounds, self.upper_bounds)
            offspring_fitness = objective_function(offspring)
            self.eval_count += self.population_size


            # Selection
            for i in range(self.population_size):
                if offspring_fitness[i] < fitness_values[i]:
                    self.population[i] = offspring[i]
                    fitness_values[i] = offspring_fitness[i]
                    if fitness_values[i] < self.best_fitness_overall:
                        self.best_fitness_overall = fitness_values[i]
                        self.best_solution_overall = self.population[i]


            # Local search around best solution
            if self.eval_count < self.budget:
              self.local_search(objective_function)
              
        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info
    
    def mutate(self, population, F, CR):
        offspring = np.copy(population)
        for i in range(self.population_size):
            r1, r2, r3 = np.random.choice(np.arange(self.population_size), 3, replace=False)
            while r1 == i or r2 == i or r3 == i: # ensure no self-selection
                r1, r2, r3 = np.random.choice(np.arange(self.population_size), 3, replace=False)
            mutant = population[r1] + F * (population[r2] - population[r3])
            cross_points = np.random.rand(self.dim) < CR
            offspring[i] = np.where(cross_points, mutant, offspring[i])
        return offspring
        
    def local_search(self, objective_function):
      if self.eval_count < self.budget:
        step_size = 0.1*(self.upper_bounds - self.lower_bounds)
        neighbor = self.best_solution_overall + np.random.uniform(-step_size, step_size)
        neighbor = np.clip(neighbor, self.lower_bounds, self.upper_bounds)
        neighbor_fitness = objective_function(neighbor.reshape(1, -1))
        self.eval_count += 1
        if neighbor_fitness[0] < self.best_fitness_overall:
          self.best_fitness_overall = neighbor_fitness[0]
          self.best_solution_overall = neighbor