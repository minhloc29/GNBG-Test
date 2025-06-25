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


import numpy as np
import random
from scipy.optimize import minimize

import numpy as np

class IslandDifferentialEvolution:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float], num_islands: int = 5, population_size: int = 20, migration_interval: int = 500, crossover_rate: float = 0.7, mutation_rate: float = 0.5):
        """
        Initializes the IslandDifferentialEvolution algorithm.

        Args:
            budget (int): Max function evaluations.
            dim (int): Problem dimensionality.
            lower_bounds (list[float]): Lower bounds for each dimension.
            upper_bounds (list[float]): Upper bounds for each dimension.
            num_islands (int): Number of independent subpopulations (islands).
            population_size (int): Number of individuals in each island.
            migration_interval (int): Number of evaluations between migrations.
            crossover_rate (float): Crossover rate for differential evolution.
            mutation_rate (float): Mutation rate (F) for differential evolution.
        """
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)
        self.num_islands = num_islands
        self.population_size = population_size
        self.migration_interval = migration_interval
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        # Initialize islands
        self.islands = []
        for _ in range(self.num_islands):
            population = np.random.uniform(self.lower_bounds, self.upper_bounds, size=(self.population_size, self.dim))
            fitnesses = np.full(self.population_size, float('inf'))
            self.islands.append({'population': population, 'fitnesses': fitnesses, 'best_solution': None, 'best_fitness': float('inf')})

    def differential_evolution(self, island_index: int, objective_function: callable) -> None:
        """
        Performs a single generation of differential evolution on a single island.

        Args:
            island_index (int): Index of the island to evolve.
            objective_function (callable): The objective function to minimize.
        """
        island = self.islands[island_index]
        population = island['population']
        fitnesses = island['fitnesses']

        # Evaluate fitness if not already evaluated
        unevaluated_indices = np.where(fitnesses == float('inf'))[0]
        if len(unevaluated_indices) > 0:
            unevaluated_solutions = population[unevaluated_indices]
            new_fitnesses = objective_function(unevaluated_solutions)
            self.eval_count += len(unevaluated_solutions)
            fitnesses[unevaluated_indices] = new_fitnesses

        for i in range(self.population_size):
            # Mutation
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = random.sample(indices, 3)

            mutant = population[a] + self.mutation_rate * (population[b] - population[c])
            mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds) # added clipping.

            # Crossover
            trial = np.copy(population[i])
            for j in range(self.dim):
                if random.random() < self.crossover_rate or j == random.randint(0, self.dim - 1):
                    trial[j] = mutant[j]

            # Selection
            trial_fitness = objective_function(np.array([trial]))[0] # Ensure objective_function receives a 2D array
            self.eval_count += 1

            if trial_fitness < fitnesses[i]:
                population[i] = trial
                fitnesses[i] = trial_fitness

                # Update best solution on this island
                if trial_fitness < island['best_fitness']:
                    island['best_solution'] = trial
                    island['best_fitness'] = trial_fitness

                # Update overall best solution
                if trial_fitness < self.best_fitness_overall:
                    self.best_solution_overall = trial
                    self.best_fitness_overall = trial_fitness
    

    def migrate(self) -> None:
         """
         Migrates individuals between islands in a ring topology.  The best individual
         from one island is sent to the next island in the ring.
         """
         best_individuals = [island['best_solution'] for island in self.islands]
         
         # Ring migration topology
         for i in range(self.num_islands):
             receiving_island_index = (i + 1) % self.num_islands
             
             # If the island has found something, overwrite a random individual in the next island.
             if best_individuals[i] is not None:
                random_index = random.randint(0, self.population_size - 1)
                self.islands[receiving_island_index]['population'][random_index] = best_individuals[i]
                self.islands[receiving_island_index]['fitnesses'][random_index] = float('inf')  # Mark as un-evaluated.
             # pass

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        """
        Optimizes the given objective function using a multi-island differential evolution algorithm.

        Args:
            objective_function (callable): The objective function to minimize.
            acceptance_threshold (float): Not used.

        Returns:
            tuple: (best_solution_1D_numpy_array, best_fitness_scalar, optimization_info_dict)
        """
        self.eval_count = 0
        self.best_solution_overall = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim) #added
        self.best_fitness_overall = float('inf')

        while self.eval_count < self.budget:
            # Evolve each island
            for i in range(self.num_islands):
                self.differential_evolution(i, objective_function)

            # Migrate individuals
            if self.eval_count % self.migration_interval == 0:
                self.migrate()

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info




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
            optimizer = IslandDifferentialEvolution(
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
            logging.error(f"Run {run + 1} failed due to: {e}", exc_info=True)
            print(f"Run {run + 1} failed: {e}")
            
if __name__ == "__main__":
    folder_path = "codes/gnbg_python"
    # Example usage
    problem_list = [16, 17, 18, 24]
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
        
        run_optimization(1000000, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate,
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