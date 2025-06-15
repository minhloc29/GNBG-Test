from scipy.io import loadmat
import os
import logging
from typing import Optional, Tuple, List
from calc_aocc_from_gnbg import calculate_aocc_from_gnbg_history
import numpy as np
import random
from codes.gnbg_python.GNBG_instances import GNBG
from datetime import datetime

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

folder = "log_test_algorithms"
os.makedirs(folder, exist_ok=True)

# Unique log filename based on timestamp
log_filename = datetime.now().strftime(f"{folder}/run_%Y%m%d_%H%M%S.log")

# Configure logging
logging.basicConfig(
    filename=log_filename,
    filemode='w',  # overwrite if exists
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


import random

class AdaptiveGaussianDEImprovedLocalSearch:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 5 * self.dim + 50
        self.F = 0.8
        self.CR = 0.9
        self.archive = []
        self.archive_size = 100
        self.C = np.eye(self.dim)
        self.F_adapt_rate = 0.05
        self.CR_adapt_rate = 0.02
        self.sigma = 0.5
        self.local_search_iterations = 10

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1, -1))[0]
        self.eval_count += 1
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
                    self.C = 0.9 * self.C + 0.1 * np.outer(best_archive_sol - np.mean([sol for sol, _ in self.archive], axis=0), best_archive_sol - np.mean([sol for sol, _ in self.archive], axis=0))
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
                        current_solution = self.best_solution_overall.copy()
                        for k in range(self.local_search_iterations):
                            neighbor = current_solution + np.random.normal(0, 0.1, self.dim)
                            neighbor = np.clip(neighbor, self.lower_bounds, self.upper_bounds)
                            neighbor_fitness = objective_function(neighbor.reshape(1, -1))[0]
                            self.eval_count += 1
                            if neighbor_fitness < self.best_fitness_overall:
                                self.best_fitness_overall = neighbor_fitness
                                self.best_solution_overall = neighbor.copy()
                                current_solution = neighbor.copy()

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


def run_optimization(MaxEvals, AcceptanceThreshold, 
                     Dimension, CompNum, MinCoordinate, MaxCoordinate,
                     CompMinPos, CompSigma, CompH, Mu, Omega,
                     Lambda, RotationMatrix, OptimumValue, OptimumPosition,
                    num_runs: int = 31,
                    seed: Optional[int] = None) -> Tuple[List[float], List[np.ndarray]]:
    """
    Run multiple optimization runs for a given problem
    
    Args:
        problem_index: GNBG problem index (1-24)
        num_runs: Number of independent runs
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (best_fitness_values, best_solutions)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Load problem instance
    from codes.gnbg_python.GNBG_instances import GNBG
    # Set up bounds
    
    # Initialize results storage
    best_values = []
    best_solutions = []
    aoccs = []
    for run in range(num_runs):
        gnbg = GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)
        bounds = (gnbg.MinCoordinate, gnbg.MaxCoordinate)
        logging.info(f"Starting run {run + 1}/{num_runs}")
        
        # Initialize algorithm
       
        
        optimizer = AdaptiveGaussianDEImprovedLocalSearch(
            budget=MaxEvals,
            dim=gnbg.Dimension,
            lower_bounds=[gnbg.MinCoordinate for _ in range(gnbg.Dimension)],
            upper_bounds=[gnbg.MaxCoordinate for _ in range(gnbg.Dimension)]
        )
        
        # Run optimization
        best_solution, best_fitness, _ = optimizer.optimize(
            objective_function=gnbg.fitness
        )
        
        aocc = calculate_aocc_from_gnbg_history(
            fe_history=gnbg.FEhistory,
            optimum_value=gnbg.OptimumValue,
            budget_B=gnbg.MaxEvals
        )
        best_values.append(best_fitness)
        best_solutions.append(best_solution)
        aoccs.append(aocc)
        
        logging.info(f"Run {run + 1} completed. Best fitness: {best_fitness:.6e}, AOCC: {aocc:.4f}")
            
    return best_values, best_solutions, aoccs
if __name__ == "__main__":
    folder_path = "codes/gnbg_python"
    # Example usage
    ProblemIndex = 1
    if 1 <= ProblemIndex <= 24:
        
        filename = f'f{ProblemIndex}.mat'
        GNBG_tmp = loadmat(os.path.join(folder_path, filename))['GNBG']
        MaxEvals = np.array([item[0] for item in GNBG_tmp['MaxEvals'].flatten()])[0, 0]
        AcceptanceThreshold = np.array([item[0] for item in GNBG_tmp['AcceptanceThreshold'].flatten()])[0, 0]
        Dimension = np.array([item[0] for item in GNBG_tmp['Dimension'].flatten()])[0, 0]
        CompNum = np.array([item[0] for item in GNBG_tmp['o'].flatten()])[0, 0]  # Number of components
        MinCoordinate = np.array([item[0] for item in GNBG_tmp['MinCoordinate'].flatten()])[0, 0]
        MaxCoordinate = np.array([item[0] for item in GNBG_tmp['MaxCoordinate'].flatten()])[0, 0]
        CompMinPos = np.array(GNBG_tmp['Component_MinimumPosition'][0, 0])
        CompSigma = np.array(GNBG_tmp['ComponentSigma'][0, 0], dtype=np.float64)
        CompH = np.array(GNBG_tmp['Component_H'][0, 0])
        Mu = np.array(GNBG_tmp['Mu'][0, 0])
        Omega = np.array(GNBG_tmp['Omega'][0, 0])
        Lambda = np.array(GNBG_tmp['lambda'][0, 0])
        RotationMatrix = np.array(GNBG_tmp['RotationMatrix'][0, 0])
        OptimumValue = np.array([item[0] for item in GNBG_tmp['OptimumValue'].flatten()])[0, 0]
        OptimumPosition = np.array(GNBG_tmp['OptimumPosition'][0, 0])
        
        best_values, best_solutions, aoccs = run_optimization(100000, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate,
                                                       CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)
    else:
        raise ValueError('ProblemIndex must be between 1 and 24.')
    
    print(f"\nResults for Problem {ProblemIndex}:")
    print(f"Best solution: {best_solutions}")
    print(f"Optimun Solution: {OptimumValue}")
    print(f"Best fitness values: {best_values[1:]}")
    print(f"Mean fitness: {np.mean(best_values[1:])}")
    print(f"Std fitness: {np.std(best_values[1:])}") 
    print(f"Mean AOCC:         {np.mean(aoccs):.4f} (Higher is better)")
    print(f"Std Dev AOCC:      {np.std(aoccs):.4f}")
    
# python test_generated_algorithm.py