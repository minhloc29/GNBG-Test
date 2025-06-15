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

class EnhancedArchiveGuidedDE:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        self.population_size = 10 * self.dim  # common heuristic
        self.archive_size = 100
        self.archive = []
        self.population = None
        self.F_scale = 0.5 #initial scaling factor


    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds, size=(self.population_size, self.dim))
        fitness = objective_function(self.population)
        self.eval_count += self.population_size

        self.best_solution_overall = self.population[np.argmin(fitness)]
        self.best_fitness_overall = np.min(fitness)

        while self.eval_count < self.budget:
            offspring = self.generate_offspring(self.population, fitness)
            offspring_fitness = objective_function(offspring)
            self.eval_count += len(offspring)

            # Update archive
            self.update_archive(offspring, offspring_fitness)

            # Select best solutions for next generation
            combined_population = np.concatenate((self.population, offspring))
            combined_fitness = np.concatenate((fitness, offspring_fitness))
            indices = np.argsort(combined_fitness)
            self.population = combined_population[indices[:self.population_size]]
            fitness = combined_fitness[indices[:self.population_size]]

            #Update best solution
            self.best_solution_overall = self.population[np.argmin(fitness)]
            self.best_fitness_overall = np.min(fitness)


        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall,
            'archive_size': len(self.archive)
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info

    def generate_offspring(self, population, fitness):
        offspring = np.zeros((self.population_size, self.dim))
        #Adaptive scaling factor
        self.F_scale = 0.5 + 0.3*np.random.rand() #scale factor with slight variation

        for i in range(self.population_size):
            # Select pbest from archive (if available)
            if self.archive:
                pbest_index = np.random.choice(len(self.archive))
                pbest = self.archive[pbest_index][0]
            else:
                pbest = population[np.argmin(fitness)]

            a, b, c = random.sample(range(self.population_size), 3)
            while a == i or b == i or c == i:
                a, b, c = random.sample(range(self.population_size), 3)

            offspring[i] = population[i] + self.F_scale * (pbest - population[i] + population[a] - population[b])
            offspring[i] = np.clip(offspring[i], self.lower_bounds, self.upper_bounds) #Boundary handling

        return offspring

    def update_archive(self, offspring, offspring_fitness):
        for i in range(len(offspring)):
            if len(self.archive) < self.archive_size:
                self.archive.append((offspring[i], offspring_fitness[i]))
            else:
                #Prioritize diversity in archive
                worst_index = np.argmax([f for _, f in self.archive])
                if offspring_fitness[i] < self.archive[worst_index][1] or len(self.archive) < self.archive_size * 0.8 :
                    self.archive[worst_index] = (offspring[i], offspring_fitness[i])


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
       
        
        optimizer = EnhancedArchiveGuidedDE(
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
    ProblemIndex = 9
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
        
        best_values, best_solutions, aoccs = run_optimization(500000, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate,
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