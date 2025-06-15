import numpy as np

class AdaptiveDE:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 10 * self.dim  # Adaptive population size
        self.F = 0.8 # Mutation factor
        self.CR = 0.9 # Crossover rate

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
            #Adaptive Mutation Strategy based on population diversity
            diversity = np.std(population, axis=0).mean()
            adaptive_F = self.F * (1 + np.exp(-diversity)) # Reduce F for high diversity, increase for low

            offspring = np.zeros_like(population)
            for i in range(self.population_size):
                #Differential Evolution Mutation
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = population[a] + adaptive_F * (population[b] - population[c])

                #Clamp mutant vector
                mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds)
                
                #Crossover
                jrand = np.random.randint(0, self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == jrand:
                        offspring[i, j] = mutant[j]
                    else:
                        offspring[i, j] = population[i, j]

            offspring_fitness = objective_function(offspring)
            self.eval_count += self.population_size

            #Selection
            for i in range(self.population_size):
                if offspring_fitness[i] < fitness_values[i]:
                    population[i] = offspring[i]
                    fitness_values[i] = offspring_fitness[i]
                    if fitness_values[i] < self.best_fitness_overall:
                        self.best_fitness_overall = fitness_values[i]
                        self.best_solution_overall = population[i, :].copy()


            #Local Search  Improved exploration radius and dynamic population size adjustment
            if self.eval_count < self.budget *0.8:
                self.population_size = int(10 * self.dim * (1 + np.exp(-self.eval_count/self.budget))) #Dynamic population size
                local_search_point = self.best_solution_overall + np.random.normal(0,0.2,self.dim) #Increased exploration radius
                local_search_point = np.clip(local_search_point, self.lower_bounds, self.upper_bounds)
                local_fitness = objective_function(np.array([local_search_point]))[0]
                self.eval_count += 1
                if local_fitness < self.best_fitness_overall:
                    self.best_fitness_overall = local_fitness
                    self.best_solution_overall = local_search_point.copy()

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info