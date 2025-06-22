from scipy.io import loadmat
import os
import logging
from typing import Optional, Tuple, List
from my_utils.utils import calculate_aocc_from_gnbg_history
import numpy as np
import random
from codes.gnbg_python.GNBG_instances import GNBG
from datetime import datetime
import random

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



class AdaptiveGaussianSamplingEAwithArchive:
    """
    Combines adaptive Gaussian sampling with an archive to enhance exploration and exploitation in multimodal landscapes.
    """
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.population_size = 100
        self.archive_size = 200
        self.sigma = 0.2 * (self.upper_bounds - self.lower_bounds)
        self.sigma_decay = 0.99
        self.archive = []

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        self.eval_count = 0
        self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        self.best_fitness_overall = objective_function(self.best_solution_overall.reshape(1, -1))[0]
        self.eval_count += 1

        population = self._initialize_population()
        fitness_values = objective_function(population)
        self.eval_count += self.population_size

        self.archive = self._update_archive(population, fitness_values)

        while self.eval_count < self.budget:
            parents = self._tournament_selection(population, fitness_values)
            offspring = self._gaussian_recombination(parents)
            offspring = self._adaptive_mutation(offspring)
            offspring_fitness = objective_function(offspring)
            self.eval_count += len(offspring)

            population, fitness_values = self._select_next_generation(
                population, fitness_values, offspring, offspring_fitness
            )

            self.archive = self._update_archive(
                np.vstack((population, offspring)),
                np.concatenate((fitness_values, offspring_fitness))
            )

            self._update_best(offspring, offspring_fitness)
            self.sigma *= self.sigma_decay

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }

        return self.best_solution_overall, self.best_fitness_overall, optimization_info

    def _initialize_population(self):
        center = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        population = np.random.normal(center, self.sigma, size=(self.population_size, self.dim))
        return np.clip(population, self.lower_bounds, self.upper_bounds)

    def _tournament_selection(self, population, fitness_values):
        tournament_size = 5
        num_parents = self.population_size // 2
        selected_parents = []

        for _ in range(num_parents):
            tournament = np.random.choice(len(population), tournament_size, replace=False)
            winner_index = tournament[np.argmin(fitness_values[tournament])]
            selected_parents.append(population[winner_index])

        return np.array(selected_parents)

    def _gaussian_recombination(self, parents):
        offspring = []

        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            midpoint = (parent1 + parent2) / 2
            child1 = midpoint + np.random.normal(0, self.sigma / 2, self.dim)
            child2 = midpoint + np.random.normal(0, self.sigma / 2, self.dim)
            offspring.extend([child1, child2])

        return np.clip(np.array(offspring), self.lower_bounds, self.upper_bounds)

    def _adaptive_mutation(self, offspring):
        mutated = offspring + np.random.normal(0, self.sigma, size=offspring.shape)
        return np.clip(mutated, self.lower_bounds, self.upper_bounds)

    def _select_next_generation(self, population, fitness_values, offspring, offspring_fitness):
        combined_pop = np.vstack((population, offspring))
        combined_fit = np.concatenate((fitness_values, offspring_fitness))
        sorted_indices = np.argsort(combined_fit)

        next_gen = combined_pop[sorted_indices[:self.population_size]]
        next_fit = combined_fit[sorted_indices[:self.population_size]]
        return next_gen, next_fit

    def _update_best(self, offspring, offspring_fitness):
        for i, fitness in enumerate(offspring_fitness):
            if fitness < self.best_fitness_overall:
                self.best_fitness_overall = fitness
                self.best_solution_overall = offspring[i]

    def _update_archive(self, population, fitness_values):
        combined = np.column_stack((population, fitness_values))
        new_archive = []

        for sol in combined:
            already_present = any(np.allclose(sol[:-1], arch[:-1], atol=1e-6) for arch in self.archive)
            if not already_present:
                new_archive.append(sol)

        new_archive.sort(key=lambda x: x[-1])
        return np.array(new_archive[:self.archive_size])



def run_optimization(MaxEvals, AcceptanceThreshold, 
                     Dimension, CompNum, MinCoordinate, MaxCoordinate,
                     CompMinPos, CompSigma, CompH, Mu, Omega,
                     Lambda, RotationMatrix, OptimumValue, OptimumPosition,
                    num_runs: int = 1,
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
       
        try:
            optimizer = AdaptiveGaussianSamplingEAwithArchive(
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
            logging.info(f"\nResults for Problem {ProblemIndex}:")
            logging.info(f"Best solution: {best_solutions}")
            logging.info(f"Optimun Solution: {OptimumValue}")
            logging.info(f"Best fitness values: {best_values}")
            logging.info(f"Mean fitness: {np.mean(best_values)}")
            logging.info(f"Std fitness: {np.std(best_values)}") 
            logging.info(f"Mean AOCC:         {np.mean(aoccs):.4f} (Higher is better)")
            logging.info(f"Std Dev AOCC:      {np.std(aoccs):.4f}")
            
        except Exception as e:
            print("Solution not found")
    return best_values, best_solutions, aoccs
if __name__ == "__main__":
    folder_path = "codes/gnbg_python"
    # Example usage
    problem_list = [1, 3, 6, 15, 22]
    for ProblemIndex in problem_list:
        
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
    
    
    # print(f"\nResults for Problem {ProblemIndex}:")
    # print(f"Best solution: {best_solutions}")
    # print(f"Optimun Solution: {OptimumValue}")
    # print(f"Best fitness values: {best_values[1:]}")
    # print(f"Mean fitness: {np.mean(best_values[1:])}")
    # print(f"Std fitness: {np.std(best_values[1:])}") 
    # print(f"Mean AOCC:         {np.mean(aoccs):.4f} (Higher is better)")
    # print(f"Std Dev AOCC:      {np.std(aoccs):.4f}")
    
# python test_generated_algorithm.py