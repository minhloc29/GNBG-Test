import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

class AdaptiveDEMGMImprovedEnhanced:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population = None
        self.fitness = None
        self.gmm = None

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1,-1))[0]
        self.eval_count +=1
        
        pop_size = max(10 * self.dim, 100) 
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, (pop_size, self.dim))
        self.fitness = objective_function(self.population)
        self.eval_count += pop_size
        
        self.best_solution_overall = self.population[np.argmin(self.fitness)]
        self.best_fitness_overall = np.min(self.fitness)
        
        self.gmm = GaussianMixture(n_components=min(pop_size//2,5), random_state=0)


        for gen in range(min(self.budget // pop_size, 100)): 
            mutated_pop = self.mutate()
            offspring = np.clip(mutated_pop, self.lower_bounds, self.upper_bounds)
            offspring_fitness = objective_function(offspring)
            self.eval_count += pop_size
            
            combined_pop = np.vstack((self.population, offspring))
            combined_fitness = np.concatenate((self.fitness, offspring_fitness))
            
            indices = np.argsort(combined_fitness)
            self.population = combined_pop[indices[:pop_size]]
            self.fitness = combined_fitness[indices[:pop_size]]

            if np.min(self.fitness) < self.best_fitness_overall:
                self.best_fitness_overall = np.min(self.fitness)
                self.best_solution_overall = self.population[np.argmin(self.fitness)]

            if self.eval_count >= self.budget:
                break
            #Enhance diversity by adding a small amount of random noise to the best solution before fitting the GMM
            self.gmm.fit(np.clip(self.population + np.random.normal(0, 0.1, size=self.population.shape), self.lower_bounds, self.upper_bounds))


        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info

    def mutate(self):
        mutated_pop = np.zeros_like(self.population)
        for i in range(len(self.population)):
            a, b, c = np.random.choice(len(self.population), 3, replace=False)
            # Modification: Weighted average for mutation vector
            v = 0.5 * self.population[a] + 0.25 * self.population[b] + 0.25 * self.population[c]
            
            means = self.gmm.means_
            covariances = self.gmm.covariances_
            weights = self.gmm.weights_

            best_mean_index = np.argmin(np.sum((means - v)**2, axis=1))
            
            #Change: Dynamic scaling factor
            F = 0.5 + 0.5 * np.random.rand() * (1 - (self.eval_count / self.budget)) #Decreases over time
            mutated_pop[i] = np.clip(multivariate_normal.rvs(mean=means[best_mean_index], cov=covariances[best_mean_index]), self.lower_bounds, self.upper_bounds)
            

        return mutated_pop