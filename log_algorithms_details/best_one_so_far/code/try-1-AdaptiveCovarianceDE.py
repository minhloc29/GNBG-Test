import numpy as np

class AdaptiveCovarianceDE:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 10 * self.dim  # Adaptive population size
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover rate
        self.population = None
        self.fitness_values = None
        self.covariance_matrix = None

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        
        #Initialization
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        self.fitness_values = objective_function(self.population)
        self.eval_count += self.population_size
        self.best_solution_overall = self.population[np.argmin(self.fitness_values)]
        self.best_fitness_overall = np.min(self.fitness_values)
        self.covariance_matrix = np.cov(self.population, rowvar=False)

        #Main Loop
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation
                a, b, c = np.random.choice(np.arange(self.population_size), 3, replace=False)
                while a == i or b == i or c == i:  #Ensure different individuals
                    a, b, c = np.random.choice(np.arange(self.population_size), 3, replace=False)

                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                
                #Adaptive Mutation using Covariance Matrix
                mutation_vector = np.random.multivariate_normal(np.zeros(self.dim), self.covariance_matrix)
                mutant += 0.2 * mutation_vector #scaling factor

                #Clipping
                mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds)
                
                #Crossover
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[i])

                #Selection
                trial_fitness = objective_function(trial.reshape(1, -1))[0]
                self.eval_count +=1
                if trial_fitness < self.fitness_values[i]:
                    self.population[i] = trial
                    self.fitness_values[i] = trial_fitness
                    if trial_fitness < self.best_fitness_overall:
                        self.best_solution_overall = trial
                        self.best_fitness_overall = trial_fitness
                        
            self.covariance_matrix = np.cov(self.population, rowvar=False) #Update Covariance Matrix

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info