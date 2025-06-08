import numpy as np
from scipy.stats import multivariate_normal

class AdaptiveGMDEImproved:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 10 * self.dim  # Adaptive population size
        self.F = 0.8  # Differential Evolution scaling factor
        self.CR = 0.9  # Crossover rate
        self.population = None
        self.fitness_values = None
        self.gmm_means = None
        self.gmm_covariances = None

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        self.fitness_values = objective_function(self.population)
        self.eval_count += self.population_size
        self.best_solution_overall = self.population[np.argmin(self.fitness_values)]
        self.best_fitness_overall = np.min(self.fitness_values)

        # Initialize GMM parameters (2 components initially)
        self.gmm_means = np.array([self.best_solution_overall, np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)])
        self.gmm_covariances = np.array([np.eye(self.dim), np.eye(self.dim)])


        while self.eval_count < self.budget:
            for i in range(self.population_size):
                # Differential Evolution Mutation
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])

                # Adaptive Gaussian Mixture Model Mutation
                if np.random.rand() < 0.3:  # Increased probability of using GMM mutation
                    weights = np.array([multivariate_normal.pdf(mutant, mean=mean, cov=cov) for mean, cov in zip(self.gmm_means, self.gmm_covariances)])
                    weights /= np.sum(weights)
                    chosen_component = np.random.choice(len(self.gmm_means), p=weights)
                    mutant = np.random.multivariate_normal(self.gmm_means[chosen_component], self.gmm_covariances[chosen_component])


                # Boundary handling
                mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds)


                # Crossover
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[i])

                # Selection
                trial_fitness = objective_function(trial.reshape(1, -1))
                self.eval_count += 1
                if trial_fitness < self.fitness_values[i]:
                    self.population[i] = trial
                    self.fitness_values[i] = trial_fitness
                    if trial_fitness < self.best_fitness_overall:
                        self.best_fitness_overall = trial_fitness
                        self.best_solution_overall = trial

            # Update GMM parameters (improved update)
            top_indices = np.argsort(self.fitness_values)[:10]  # Take top 10 diverse solutions
            self.gmm_means = self.population[top_indices]
            self.gmm_covariances = np.array([np.cov(self.population[idx].reshape(1,-1).T) + 0.1 * np.eye(self.dim) for idx in top_indices]) #Regularization


        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info