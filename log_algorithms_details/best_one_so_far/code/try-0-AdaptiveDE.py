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
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover rate
        self.population = None
        self.fitness = None

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        if self.dim > 0:
            self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
            self.fitness = objective_function(self.population)
            self.eval_count += self.population_size
        else:
            self.population = np.array([])
            self.fitness = np.array([])
        self.best_solution_overall = self.population[np.argmin(self.fitness)]
        self.best_fitness_overall = np.min(self.fitness)

        while self.eval_count < self.budget:
            new_population = np.zeros_like(self.population)
            for i in range(self.population_size):
                # Select parents
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                
                # Adaptive Mutation
                diff = self.population[a] - self.population[b]
                gradient_estimate = np.zeros(self.dim)
                
                # Simple gradient estimate based on neighboring solutions
                if i>0:
                    gradient_estimate = (self.population[i] - self.population[i-1])/(self.fitness[i] - self.fitness[i-1]) if self.fitness[i] != self.fitness[i-1] else np.zeros(self.dim)

                mutated = self.population[i] + self.F * (self.population[c] + gradient_estimate)
                mutated = np.clip(mutated, self.lower_bounds, self.upper_bounds)
                
                # Crossover
                j_rand = np.random.randint(0, self.dim)
                trial = np.copy(self.population[i])
                trial[np.random.rand(self.dim) < self.CR] = mutated[np.random.rand(self.dim) < self.CR]
                trial[j_rand] = mutated[j_rand]

                trial_fitness = objective_function(trial.reshape(1, -1))[0]
                self.eval_count += 1
                if trial_fitness < self.fitness[i]:
                    new_population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness_overall:
                        self.best_fitness_overall = trial_fitness
                        self.best_solution_overall = trial
                else:
                    new_population[i] = self.population[i]

            self.population = new_population

        if self.best_solution_overall is None and self.dim > 0:
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info