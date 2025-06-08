import numpy as np

class AdaptiveDEImproved2:
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
        self.successful_solutions = [] #Archive of successful solutions
        self.success_weight = 0.5 #Weight given to successful solutions


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

                #Adaptive Mutation using covariance matrix and successful solutions
                if self.successful_solutions:
                    weighted_mean = self.success_weight * np.mean(self.successful_solutions, axis=0) + (1-self.success_weight) * mutant
                    mutated_vector = np.random.multivariate_normal(weighted_mean, self.covariance_matrix)
                else:
                    mutated_vector = np.random.multivariate_normal(mutant, self.covariance_matrix)

                mutated_vector = np.clip(mutated_vector, self.lower_bounds, self.upper_bounds)

                #Diversity Promoting Mutation (small random perturbation)
                mutated_vector += np.random.normal(0, 0.1, self.dim) #Added line
                mutated_vector = np.clip(mutated_vector, self.lower_bounds, self.upper_bounds) #Added line

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
                        self.successful_solutions.append(trial)

            #Covariance Matrix Adaptation
            mean_diff = np.mean(self.population - np.mean(self.population, axis=0),axis=0)
            self.covariance_matrix = (1 - self.learning_rate) * self.covariance_matrix + self.learning_rate * np.outer(mean_diff, mean_diff)
            self.covariance_matrix = (self.covariance_matrix + self.covariance_matrix.T) / 2 #ensure symmetry

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info