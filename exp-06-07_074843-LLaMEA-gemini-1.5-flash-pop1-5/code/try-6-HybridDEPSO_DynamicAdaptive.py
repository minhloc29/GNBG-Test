import numpy as np
from scipy.optimize import minimize

class HybridDEPSO_DynamicAdaptive:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None 
        self.best_fitness_overall = float('inf')
        self.population_size = int(np.sqrt(self.budget)) #Heuristic population size
        self.F = 0.8 # Mutation factor for DE
        self.CR = 0.9 # Crossover rate for DE
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.4

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        if self.dim > 0:
            self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
            self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1,-1))[0]
            self.eval_count +=1
        else:
            self.best_solution_overall = np.array([])

        population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        velocities = np.zeros_like(population)
        personal_bests = population.copy()
        personal_best_fitnesses = np.apply_along_axis(lambda x: objective_function(x.reshape(1,-1))[0], 1, population)
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            #Dynamic population size adjustment
            exploration_rate = 1 - (self.eval_count / self.budget)
            self.population_size = int(np.sqrt(self.budget) * (0.5 + 0.5 * exploration_rate)) # Adjust population size dynamically

            #Differential Evolution
            for i in range(self.population_size):
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                trial_fitness = objective_function(trial.reshape(1,-1))[0]
                self.eval_count +=1
                if trial_fitness < personal_best_fitnesses[i]:
                    personal_bests[i] = trial
                    personal_best_fitnesses[i] = trial_fitness
                    if trial_fitness < self.best_fitness_overall:
                        self.best_solution_overall = trial
                        self.best_fitness_overall = trial_fitness

            #Particle Swarm Optimization
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                #Adaptive inertia weight and mutation factor
                self.inertia_weight = 0.4 + 0.3 * np.exp(-self.eval_count / self.budget)
                self.F = 0.5 + 0.3 * np.exp(-self.eval_count / self.budget) #Adaptive mutation
                velocities[i] = self.inertia_weight * velocities[i] + self.cognitive_coeff * r1 * (personal_bests[i] - population[i]) + self.social_coeff * r2 * (self.best_solution_overall - population[i])
                population[i] = population[i] + velocities[i]
                population[i] = np.clip(population[i], self.lower_bounds, self.upper_bounds)

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info