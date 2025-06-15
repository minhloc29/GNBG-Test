import numpy as np
from scipy.stats import multivariate_normal

class AdaptiveGaussianDE:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 5 * self.dim + 50 # Refined population sizing
        self.F = 0.8
        self.CR = 0.9
        self.archive = []
        self.archive_size = 100
        self.C = np.eye(self.dim)
        self.F_adapt_rate = 0.05
        self.CR_adapt_rate = 0.02
        self.sigma = 0.5


    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1,-1))[0]
        self.eval_count +=1
        population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        fitness_values = objective_function(population)
        self.eval_count += self.population_size

        for i in range(self.budget - self.population_size):
            mutated = np.zeros((self.population_size, self.dim))
            for j in range(self.population_size):
                a, b, c = np.random.choice(np.arange(self.population_size), size=3, replace=False)
                while a == j or b == j or c == j:
                    a, b, c = np.random.choice(np.arange(self.population_size), size=3, replace=False)
                mutated[j] = population[a] + self.F * (population[b] - population[c])

            crossed = np.zeros((self.population_size, self.dim))
            for j in range(self.population_size):
                rand = np.random.rand(self.dim)
                crossed[j] = np.where(rand < self.CR, mutated[j], population[j])

            if len(self.archive) > self.dim and len(self.archive) > 0:
                from sklearn.mixture import GaussianMixture
                gm = GaussianMixture(n_components=min(len(self.archive), 10), covariance_type='full', random_state=0)
                gm.fit(np.array([sol for sol, _ in self.archive]))
                new_mutations = gm.sample(self.population_size)[0]
                new_mutations = np.clip(new_mutations, self.lower_bounds, self.upper_bounds)
                new_mutations = np.random.multivariate_normal(np.zeros(self.dim), self.C * (self.sigma**2), self.population_size)
                crossed = np.clip(crossed + new_mutations * 0.2, self.lower_bounds, self.upper_bounds)
                
                if len(self.archive) > 0:
                    best_archive_sol = np.array([sol for sol, _ in self.archive])[np.argmin([f for _, f in self.archive])]
                    self.C = 0.9 * self.C + 0.1 * np.outer(best_archive_sol - np.mean([sol for sol,_ in self.archive], axis=0),best_archive_sol - np.mean([sol for sol,_ in self.archive], axis=0))
                    self.sigma *= 0.99

            offspring_fitness = objective_function(crossed)
            self.eval_count += self.population_size
            for j in range(self.population_size):
                if offspring_fitness[j] < fitness_values[j]:
                    fitness_values[j] = offspring_fitness[j]
                    population[j] = crossed[j]
                    if offspring_fitness[j] < self.best_fitness_overall:
                        self.best_fitness_overall = offspring_fitness[j]
                        self.best_solution_overall = crossed[j]
                        
            for j in range(self.population_size):
              if len(self.archive) < self.archive_size:
                  self.archive.append((population[j], fitness_values[j]))
              else:
                  if fitness_values[j] < np.max([f for _, f in self.archive]):
                      self.archive.pop(np.argmax([f for _, f in self.archive]))
                      self.archive.append((population[j], fitness_values[j]))
            self.F = self.F * (1 + self.F_adapt_rate * np.random.randn())
            self.CR = self.CR * (1 + self.CR_adapt_rate * np.random.randn())
            self.F = np.clip(self.F, 0.1, 1)
            self.CR = np.clip(self.CR, 0.1, 1)

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info