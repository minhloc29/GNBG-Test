import numpy as np
from scipy.optimize import minimize

class HybridPSO_NelderMead:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None 
        self.best_fitness_overall = float('inf')
        self.swarm_size = int(np.sqrt(self.budget)) #Heuristic swarm size
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

        swarm = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.swarm_size, self.dim))
        velocities = np.zeros_like(swarm)
        personal_bests = swarm.copy()
        personal_best_fitnesses = np.apply_along_axis(lambda x: objective_function(x.reshape(1,-1))[0], 1, swarm)
        self.eval_count += self.swarm_size

        while self.eval_count < self.budget:
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = self.inertia_weight * velocities[i] + self.cognitive_coeff * r1 * (personal_bests[i] - swarm[i]) + self.social_coeff * r2 * (self.best_solution_overall - swarm[i])
                swarm[i] = swarm[i] + velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bounds, self.upper_bounds)

            fitness_values = objective_function(swarm)
            self.eval_count += self.swarm_size
            for i in range(self.swarm_size):
                if fitness_values[i] < personal_best_fitnesses[i]:
                    personal_bests[i] = swarm[i]
                    personal_best_fitnesses[i] = fitness_values[i]
                    if fitness_values[i] < self.best_fitness_overall:
                        self.best_solution_overall = swarm[i]
                        self.best_fitness_overall = fitness_values[i]
                        
            # Local search with Nelder-Mead for the best solution
            res = minimize(objective_function, self.best_solution_overall, method='Nelder-Mead', options={'maxfev': min(100,self.budget - self.eval_count)})
            if res.fun < self.best_fitness_overall and self.eval_count + res.nfev <= self.budget:
                self.best_fitness_overall = res.fun
                self.best_solution_overall = res.x
                self.eval_count += res.nfev
            else:
                break

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info
