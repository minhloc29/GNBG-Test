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
        self.population_size = 10 * self.dim # Rule of thumb
        self.F = 0.8 # DE scaling factor
        self.CR = 0.9 # DE crossover rate
        self.covariance_matrix = np.eye(self.dim) # Initial covariance matrix
        self.learning_rate = 0.1 # Learning rate for covariance matrix adaptation
        self.archive = [] # Archive of good solutions
        self.archive_size = 100
        self.stagnation_counter = 0
        self.stagnation_threshold = 10

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        fitness_values = objective_function(population)
        self.eval_count += self.population_size

        best_index = np.argmin(fitness_values)
        self.best_solution_overall = population[best_index].copy()
        self.best_fitness_overall = fitness_values[best_index]

        previous_best = self.best_fitness_overall
        for _ in range(self.budget // self.population_size):
            offspring = []
            for i in range(self.population_size):
                # Differential Evolution mutation
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])

                # Adaptive mutation using covariance matrix
                mutation_vector = np.random.multivariate_normal(np.zeros(self.dim), self.covariance_matrix)
                mutant += (self.F * (1 + 0.1 * np.random.rand()) * mutation_vector) #Diversity based scaling


                # Clip to bounds
                mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])

                offspring.append(trial)

            offspring = np.array(offspring)
            offspring_fitness = objective_function(offspring)
            self.eval_count += self.population_size


            for i in range(self.population_size):
                if offspring_fitness[i] < fitness_values[i]:
                    population[i] = offspring[i]
                    fitness_values[i] = offspring_fitness[i]
                    if fitness_values[i] < self.best_fitness_overall:
                        self.best_solution_overall = offspring[i].copy()
                        self.best_fitness_overall = fitness_values[i]
                        self.stagnation_counter = 0 #reset stagnation counter
                    self.archive.append((offspring[i], offspring_fitness[i]))
                    self.archive = sorted(self.archive, key=lambda item: item[1])[:self.archive_size]
                
            if self.best_fitness_overall == previous_best:
                self.stagnation_counter +=1
            else:
                previous_best = self.best_fitness_overall
                
            #local search triggered by stagnation
            if self.stagnation_counter >= self.stagnation_threshold:
                # Implement local search here (e.g., Nelder-Mead)
                pass # placeholder for local search


            # Covariance matrix adaptation
            selected_solutions = np.array([sol for sol, fit in self.archive])
            if selected_solutions.shape[0] > 0:
                mean = np.mean(selected_solutions, axis=0)
                cov = np.cov(selected_solutions, rowvar=False)
                self.covariance_matrix = (1 - self.learning_rate) * self.covariance_matrix + self.learning_rate * cov

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info