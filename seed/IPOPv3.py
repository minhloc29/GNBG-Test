from scipy.io import loadmat
import os
from test.call_llm import RealLLM
import logging
import cma
from typing import Optional, Tuple, List
from my_utils.utils import calculate_aocc_from_gnbg_history
import numpy as np
import cma
import random
from codes.gnbg_python.GNBG_instances import GNBG
from datetime import datetime
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
# from seed.IPOPCMAES import IPOPCMAES
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

from scipy.optimize import minimize

# Name: AdaptiveIslandDE
# Description: An island-based DE with adaptive migration, local search, and restart mechanisms to handle deceptive landscapes.
# Code:
# Terrible at: f9, f24, f13, f14, f15, f21, f22
from joblib import Parallel, delayed

import numpy as np
import logging
import random
import cma
def parallel_llm_generate(llm, good, bad, batch_size, num_batches):
    """
    Generates LLM candidates in parallel using multiple CPU cores.
    """
    results = Parallel(n_jobs=2)(delayed(llm.fine_tune_and_generate)(
        good, bad, batch_size) for _ in range(num_batches))
    
    # Flatten list of lists
    return [x for batch in results for x in batch]
class IPOPCMAES_LLM:
    """
    Implements the IPOP-CMA-ES algorithm with an integrated Large Language Model (LLM)
    to enhance the evolutionary search, particularly for accelerating convergence.
    The LLM acts as an intelligent operator, generating promising candidate solutions
    based on the history of evaluated solutions.
    """
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float],
                 initial_population_size: int = 100, initial_sigma: float = 5.0,
                 max_restarts: int = 50, population_size_multiplier: float = 3.0,
                 llm_influence_ratio: float = 0.5, # Ratio of LLM's contribution to starting points / candidate pool
                 llm_prompt_history_size: int = 200, # Number of good/bad solutions to feed to LLM
                 llm_generate_count_per_call: int = 5 # Number of solutions LLM generates in one go
                 ):
        """
        Initializes the IPOPCMAES optimizer with LLM integration.

        Args:
            budget (int): Maximum total number of function evaluations allowed across all restarts.
            dim (int): Dimensionality of the problem.
            lower_bounds (list[float]): List of lower bounds for each dimension of the decision variables.
            upper_bounds (list[float]): List of upper bounds for each dimension of the decision variables.
            initial_population_size (int): Initial population size for CMA-ES.
                                           If 0 or None, CMA-ES default (4 + floor(3*log(dim))) is used.
            initial_sigma (float): Initial standard deviation for the CMA-ES search distribution.
            max_restarts (int): Maximum number of restarts allowed for the IPOP-CMA-ES strategy.
            population_size_multiplier (float): Factor by which the population size increases on each restart.
            llm_influence_ratio (float): A conceptual ratio indicating how much the LLM's candidates are considered.
                                         In this implementation, it's used to decide if the LLM seeds the next x0.
            llm_prompt_history_size (int): The number of best and worst solutions from the `solution_history`
                                           to be passed to the LLM for its "prompt engineering."
            llm_generate_count_per_call (int): The number of new candidate solutions the LLM attempts to generate
                                               each time it's invoked.
        """
        # Core Optimization Parameters
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)
        # Use a default if initial_population_size is explicitly set to 0 or None
        self.initial_population_size = initial_population_size if initial_population_size > 0 else (4 + int(np.floor(3 * np.log(dim))))
        self.initial_sigma = initial_sigma
        self.max_restarts = max_restarts
        self.population_size_multiplier = population_size_multiplier

        # LLM Integration Parameters
        self.llm_influence_ratio = llm_influence_ratio
        self.llm_prompt_history_size = llm_prompt_history_size
        self.llm_generate_count_per_call = llm_generate_count_per_call
        self.llm = RealLLM(self.dim, self.lower_bounds, self.upper_bounds) # Initialize the LLM mock

        # Optimization State Variables
        self.eval_count = 0                     # Total function evaluations used
        self.best_solution_overall = None       # Best decision variable vector found so far
        self.best_fitness_overall = float('inf')# Best fitness value found so far (minimization)
        self.solution_history = []              # Stores all evaluated solutions for LLM's learning:
                                                # [{'decs': list, 'objs': float, 'cv': float (for CMOP)}]

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8, optimum_value=None) -> tuple:
        """
        Executes the IPOP-CMA-ES optimization process, integrated with an LLM that acts as an offspring generator.

        Args:
            objective_function (callable): The function to be minimized. It should accept a 2D numpy array
                                        (batch_size, dim) and return a 1D numpy array of fitness values.
            acceptance_threshold (float): The threshold for fitness value difference from optimum_value
                                        to consider the optimum reached.
            optimum_value (float, optional): The known optimal fitness value for early stopping.

        Returns:
            tuple: (best_solution, best_fitness, optimization_info_dict)
                best_solution (np.ndarray): The decision variables of the best solution found.
                best_fitness (float): The fitness value of the best solution found.
                optimization_info_dict (dict): A dictionary containing optimization statistics.
        """
        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.solution_history = []

        current_population_size = self.initial_population_size
        num_restarts = 0

        while self.eval_count < self.budget and num_restarts <= self.max_restarts:
            logging.info(f"IPOP-CMA-ES-LLM: Restart {num_restarts + 1}, pop size: {current_population_size}, FEs: {self.eval_count}/{self.budget}")

            # Step 1: CMA-ES execution
            cma_options = {
                'bounds': [list(self.lower_bounds), list(self.upper_bounds)],
                'popsize': current_population_size,
                'maxfevals': self.budget - self.eval_count,
                'verb_disp': 0,
                'seed': np.random.randint(0, 1_000_000),
            }

            def cma_objective_wrapper(x):
                if self.eval_count >= self.budget:
                    return float('inf')
                fitness_val = objective_function(x.reshape(1, -1))[0]
                self.eval_count += 1
                self.solution_history.append({'decs': x.tolist(), 'objs': fitness_val, 'cv': 0.0})
                if fitness_val < self.best_fitness_overall:
                    self.best_fitness_overall = fitness_val
                    self.best_solution_overall = x.copy()
                    logging.info(f"CMA-ES improved best fitness: {fitness_val:.6e}")
                return fitness_val

            x0 = (self.lower_bounds + self.upper_bounds) / 2.0
            try:
                result = cma.fmin(cma_objective_wrapper, x0, self.initial_sigma, cma_options)
            except Exception as e:
                logging.error(f"CMA-ES failed: {e}", exc_info=True)
                break

            # Step 2: LLM Offspring Generation
            llm_candidate_solutions = []
            if len(self.solution_history) >= self.llm_prompt_history_size:
                sorted_history = sorted(self.solution_history, key=lambda s: s['objs'])
                good = sorted_history[:self.llm_prompt_history_size]
                bad = sorted_history[-self.llm_prompt_history_size:]
                total_llm = int(current_population_size * self.llm_influence_ratio)
                batch_size = self.llm_generate_count_per_call
                num_batches = (total_llm + batch_size - 1) // batch_size
                llm_candidate_solutions = parallel_llm_generate(
                    self.llm, good, bad, batch_size=batch_size, num_batches=num_batches
                )
                logging.info(f"LLM generated {len(llm_candidate_solutions)} offspring.")

            # Step 3: Evaluate and merge offspring
            for dec in llm_candidate_solutions:
                if self.eval_count >= self.budget:
                    break
                fitness_val = objective_function(dec.reshape(1, -1))[0]
                self.eval_count += 1
                self.solution_history.append({'decs': dec.tolist(), 'objs': fitness_val, 'cv': 0.0})
                if fitness_val < self.best_fitness_overall:
                    self.best_fitness_overall = fitness_val
                    self.best_solution_overall = dec.copy()
                    logging.info(f"LLM offspring improved best fitness: {fitness_val:.6e}")

            if self.eval_count >= self.budget or (
                optimum_value is not None and abs(self.best_fitness_overall - optimum_value) <= acceptance_threshold):
                logging.info("Terminating: Budget exhausted or optimum reached.")
                break

            num_restarts += 1
            current_population_size = int(current_population_size * self.population_size_multiplier)
            current_population_size = min(current_population_size, 100 * self.dim)

        return self.best_solution_overall, self.best_fitness_overall, {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall,
            'num_restarts_performed': num_restarts
        }





def run_optimization(MaxEvals, AcceptanceThreshold, 
                     Dimension, CompNum, MinCoordinate, MaxCoordinate,
                     CompMinPos, CompSigma, CompH, Mu, Omega,
                     Lambda, RotationMatrix, OptimumValue, OptimumPosition,
                    num_runs: int = 15,
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
    base_seed = 42
    
    # Load problem instance
    from codes.gnbg_python.GNBG_instances import GNBG
    # Set up bounds
    
    # Initialize results storage
    best_solutions = []
    aoccs = []
    best_values = []
    best_params = []
    value_filename = os.path.join(folder_path, f"f_{ProblemIndex}_value.txt")
    param_filename = os.path.join(folder_path, f"f_{ProblemIndex}_param.txt")
    
    with open(value_filename, 'w') as vf, open(param_filename, 'w') as pf:
   
        for run in range(num_runs):
            run_seed = base_seed + run + 10  # Vary seed per run
            np.random.seed(run_seed)
            random.seed(run_seed)
            gnbg = GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)
            bounds = (gnbg.MinCoordinate, gnbg.MaxCoordinate)
            logging.info(f"Starting run {run + 1}/{num_runs}")
            
            # Initialize algorithm
            
            try:
                optimizer = IPOPCMAES_LLM(
                    budget=MaxEvals,
                    dim=gnbg.Dimension,
                    lower_bounds=[gnbg.MinCoordinate for _ in range(gnbg.Dimension)],
                    upper_bounds=[gnbg.MaxCoordinate for _ in range(gnbg.Dimension)]
                )
                
                # Run optimization
                best_solution, best_fitness, _ = optimizer.optimize(
                    objective_function=gnbg.fitness,
                    optimum_value = gnbg.OptimumValue
                )
                best_values.append(best_fitness)
                best_params.append(best_solution)
                
                aocc = calculate_aocc_from_gnbg_history(
                    fe_history=gnbg.FEhistory,
                    optimum_value=gnbg.OptimumValue,
                    budget_B=gnbg.MaxEvals
                )
                best_solutions.append(best_solution)
                aoccs.append(aocc)
                
                logging.info(f"Run {run + 1} completed. Best fitness: {best_fitness:.6e}, AOCC: {aocc:.4f}")
                logging.info(f"\nResults for Problem {ProblemIndex}:")
                logging.info(f"History: {gnbg.FEhistory}")
                logging.info(f"Best solution: {best_solutions}")
                logging.info(f"Optimun Solution: {OptimumValue}")
                logging.info(f"Best fitness values: {best_values}")
                logging.info(f"Mean fitness: {np.mean(best_values)}")
                logging.info(f"Std fitness: {np.std(best_values)}") 
                logging.info(f"Mean AOCC:         {np.mean(aoccs):.4f} (Higher is better)")
                logging.info(f"Std Dev AOCC:      {np.std(aoccs):.4f}")
                
                pf.write(','.join(map(str, best_solution)) + "\n")
                vf.write(str(abs(best_fitness - gnbg.OptimumValue)) + "\n")
            except Exception as e:
                logging.info(f"Run {run + 1} failed due to: {e}", exc_info=True)
                print(f"Run {run + 1} failed: {e}")
            
            convergence = []
            best_error = float('inf')
            for value in gnbg.FEhistory:
                error = abs(value - OptimumValue)
                if error < best_error:
                    best_error = error
                convergence.append(best_error)

            # Plotting the convergence
            plt.plot(range(1, len(convergence) + 1), convergence)
            plt.xlabel('Function Evaluation Number (FE)')
            plt.ylabel('Error')
            plt.title('Convergence Plot')
            plt.yscale('log')  # Set y-axis to logarithmic scale  
            plt.ylim(bottom=1e-8)  # Set lower limit to 10^-8
            plt.tight_layout()
            plt.savefig(f"{folder}/convergence_problem{ProblemIndex}_run{run + 1}.png")
            plt.close()
            
        

if __name__ == "__main__":
    folder_path = "codes/gnbg_python"
    # Example usage
    problem_list = [13]
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
        
        run_optimization(500000, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate,
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