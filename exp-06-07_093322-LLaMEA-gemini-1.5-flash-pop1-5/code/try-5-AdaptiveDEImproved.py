import numpy as np

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
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover rate
        self.covariance_matrix = np.eye(self.dim)  # Initialize covariance matrix
        self.learning_rate = 0.1 # Learning rate for covariance matrix adaptation
        self.diversity_threshold = 0.1 #Threshold for diversity control


    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        if self.dim > 0:
            self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        else:
            self.population = np.array([])
        fitness_values = objective_function(self.population)
        self.eval_count += self.population_size
        best_index = np.argmin(fitness_values)
        self.best_solution_overall = self.population[best_index].copy()
        self.best_fitness_overall = fitness_values[best_index]


        while self.eval_count < self.budget:
            for i in range(self.population_size):
                #Differential Evolution Mutation
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])

                #Adaptive Mutation using covariance matrix
                mutated_vector = np.random.multivariate_normal(mutant, self.covariance_matrix)
                mutated_vector = np.clip(mutated_vector, self.lower_bounds, self.upper_bounds)

                #Crossover
                trial = np.where(np.random.rand(self.dim) < self.CR, mutated_vector, self.population[i])

                #Selection
                trial_fitness = objective_function(trial.reshape(1, -1))
                self.eval_count += 1
                if trial_fitness[0] < fitness_values[i]:
                    self.population[i] = trial
                    fitness_values[i] = trial_fitness[0]
                    if trial_fitness[0] < self.best_fitness_overall:
                        self.best_fitness_overall = trial_fitness[0]
                        self.best_solution_overall = trial.copy()

            #Covariance Matrix Adaptation with rank-based update
            ranked_indices = np.argsort(fitness_values)
            top_individuals = self.population[ranked_indices[:int(0.2*self.population_size)]] #consider top 20%
            mean_diff = np.mean(top_individuals - np.mean(top_individuals, axis=0),axis=0)
            self.covariance_matrix = (1 - self.learning_rate) * self.covariance_matrix + self.learning_rate * np.outer(mean_diff, mean_diff)
            self.covariance_matrix = (self.covariance_matrix + self.covariance_matrix.T) / 2 #ensure symmetry

            #Diversity Control: If diversity is too low, increase exploration
            diversity = np.mean(np.std(self.population, axis=0))
            if diversity < self.diversity_threshold:
                self.F = min(1.5, self.F + 0.1) #increase mutation strength
                self.CR = min(1.0, self.CR + 0.1) #increase crossover rate


        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info