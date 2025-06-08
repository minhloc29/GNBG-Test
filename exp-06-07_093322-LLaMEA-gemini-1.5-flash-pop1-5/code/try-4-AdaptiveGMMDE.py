import numpy as np
from scipy.stats import multivariate_normal

class AdaptiveGMMDE:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 10 * self.dim  # Adaptive population size
        self.population = None
        self.fitness_values = None
        self.F = 0.8 # Differential Evolution scaling factor
        self.CR = 0.9 # Crossover rate
        self.gmm_components = 3 # Number of GMM components

    def initialize_population(self):
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        self.fitness_values = np.full(self.population_size, np.inf)

    def bound_solution(self, solution):
        solution = np.clip(solution, self.lower_bounds, self.upper_bounds)
        return solution

    def differential_evolution_step(self, individual_index):
        a, b, c = np.random.choice(np.delete(np.arange(self.population_size), individual_index), 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        mutant = self.bound_solution(mutant)
        trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[individual_index])
        return trial

    def gmm_adaptive_mutation(self, individual):
        if self.eval_count > self.population_size: # only after initial evaluation
            X = self.population[self.fitness_values < np.inf] # Avoid NaN
            if X.shape[0] > self.gmm_components: # prevent errors in fit
                gmm = GaussianMixture(n_components=self.gmm_components, covariance_type='full', random_state=0).fit(X)
                means = gmm.means_
                covariances = gmm.covariances_
                weights = gmm.weights_

                component_index = np.random.choice(self.gmm_components, p=weights)
                new_point = multivariate_normal.rvs(mean=means[component_index], cov=covariances[component_index])
                new_point = self.bound_solution(new_point)

                return new_point
            else: return individual
        else: return individual

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.initialize_population()

        for i in range(self.population_size):
            if self.eval_count < self.budget:
                fitness = objective_function(self.population[i:i+1, :])[0]
                self.fitness_values[i] = fitness
                self.eval_count += 1
                if fitness < self.best_fitness_overall:
                    self.best_fitness_overall = fitness
                    self.best_solution_overall = self.population[i, :].copy()

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count < self.budget:
                    trial_vector = self.differential_evolution_step(i)
                    trial_vector = self.gmm_adaptive_mutation(trial_vector) # Adaptive mutation
                    trial_fitness = objective_function(trial_vector.reshape(1,-1))[0]
                    self.eval_count += 1

                    if trial_fitness < self.fitness_values[i]:
                        self.fitness_values[i] = trial_fitness
                        self.population[i, :] = trial_vector.copy()
                        if trial_fitness < self.best_fitness_overall:
                            self.best_fitness_overall = trial_fitness
                            self.best_solution_overall = trial_vector.copy()

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info

from sklearn.mixture import GaussianMixture
