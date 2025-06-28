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
import matplotlib.pyplot as plt
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

import numpy as np
import random

# Name: AdaptiveIslandDE
# Description: An island-based DE with adaptive migration, local search, and restart mechanisms to handle deceptive landscapes.
# Code:
# Terrible at: f9, f24, f13, f14, f15, f21, f22
import numpy as np
import random
import logging

class AdaptiveIslandDE:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float],
                 num_islands: int = 5, population_size: int = 20, crossover_rate: float = 0.8719572569354708,
                 mutation_rate: float = 0.6113964692124271, migration_interval: int = 896, migration_size: int = 2,
                 local_search_iterations: int = 6, local_search_perturbation_scale: float = 0.1446,
                 stagnation_threshold: float = 1e-2, stagnation_patient: int = 100,
                 adaptive_stagnation: bool = True, initial_stagnation_threshold: float = 1e-2, final_stagnation_threshold: float = 1e-6):
        # Initialization of parameters
        self.initial_stagnation_threshold = initial_stagnation_threshold
        self.final_stagnation_threshold = final_stagnation_threshold
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)
        self.num_islands = num_islands
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.migration_interval = int(migration_interval)
        self.migration_size = int(migration_size)
        self.local_search_iterations = int(local_search_iterations)
        self.local_search_perturbation_scale = local_search_perturbation_scale
        self.adaptive_stagnation = adaptive_stagnation
        self.stagnation_patience = stagnation_patient
        self.stagnation_threshold = stagnation_threshold

        # State variables
        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.global_fitness_history = []
        self.last_global_restart_fe = -float('inf')

        # Initialize populations
        self.populations = [
            np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
            for _ in range(self.num_islands)
        ]
        self.fitness_values = [np.full(self.population_size, np.inf) for _ in range(self.num_islands)]
        self.best_solutions = [None] * self.num_islands
        self.best_fitnesses = [np.inf] * self.num_islands
        self.fitness_history = [[] for _ in range(self.num_islands)]

    def biased_restart(self, region: str = "positive"):
        if region == "positive":
            return np.random.uniform(70, 90, (self.population_size, self.dim))
        elif region == "negative":
            return np.random.uniform(-90, -70, (self.population_size, self.dim))
        else:
            return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
    def differential_evolution_step(self, island_index: int, objective_function: callable):
        pop = self.populations[island_index]
        fit_vals = self.fitness_values[island_index]
        for i in range(self.population_size):
            idxs = list(range(self.population_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            mutant = pop[a] + self.mutation_rate * (pop[b] - pop[c])
            mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds)

            trial = pop[i].copy()
            for d in range(self.dim):
                if random.random() < self.crossover_rate:
                    trial[d] = mutant[d]

            fitness = objective_function(trial.reshape(1, -1))[0]
            self.eval_count += 1
            if fitness < fit_vals[i]:
                pop[i] = trial
                fit_vals[i] = fitness
                if fitness < self.best_fitnesses[island_index]:
                    self.best_fitnesses[island_index] = fitness
                    self.best_solutions[island_index] = trial
                if fitness < self.best_fitness_overall:
                    self.best_fitness_overall = fitness
                    self.best_solution_overall = trial

        self.populations[island_index] = pop
        self.fitness_values[island_index] = fit_vals

    def local_search(self, solution: np.ndarray, objective_function: callable) -> tuple:
        best = solution.copy()
        best_fit = objective_function(best.reshape(1, -1))[0]
        self.eval_count += 1
        for _ in range(self.local_search_iterations):
            scale = self.local_search_perturbation_scale * np.random.uniform(0.5, 2.0)
            trial = np.clip(best + np.random.normal(0, scale, self.dim), self.lower_bounds, self.upper_bounds)
            fit = objective_function(trial.reshape(1, -1))[0]
            self.eval_count += 1
            if fit < best_fit:
                best_fit = fit
                best = trial
        return best, best_fit

    def migrate(self, objective_function: callable):
        for i in range(self.num_islands):
            dest = random.choice([j for j in range(self.num_islands) if j != i])
            best_idxs = np.argsort(self.fitness_values[i])[:self.migration_size]
            migrants = self.populations[i][best_idxs].copy()
            worst_idxs = np.argsort(self.fitness_values[dest])[-self.migration_size:]
            self.populations[dest][worst_idxs] = migrants
            new_fits = []
            for j, w in enumerate(worst_idxs):
                sol, fit = self.local_search(migrants[j], objective_function)
                self.populations[dest][w] = sol
                self.fitness_values[dest][w] = fit
                self.eval_count += 1
                if fit < self.best_fitnesses[dest]:
                    self.best_fitnesses[dest] = fit
                    self.best_solutions[dest] = sol
                if fit < self.best_fitness_overall:
                    self.best_fitness_overall = fit
                    self.best_solution_overall = sol

    def detect_stagnation(self, fitness_history: list[float]) -> bool:
        if len(fitness_history) < self.stagnation_patience:
            return False
        recent = fitness_history[-self.stagnation_patience:]
        std_r = np.std(recent)
        imp = abs(recent[0] - recent[-1])
        if self.adaptive_stagnation:
            log_s = np.log10(self.initial_stagnation_threshold)
            log_e = np.log10(self.final_stagnation_threshold)
            prog = self.eval_count / self.budget
            thr = 10 ** (log_s + prog * (log_e - log_s))
        else:
            thr = self.initial_stagnation_threshold
        return std_r < thr and imp < thr

    def restart_island(self, i: int, objective_function: callable):
        logging.info(f"Restarting island {i} at FE={self.eval_count}")
        self.populations[i] = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        self.fitness_values[i] = objective_function(self.populations[i])
        self.eval_count += self.population_size
        idx = np.argmin(self.fitness_values[i])
        self.best_fitnesses[i] = self.fitness_values[i][idx]
        self.best_solutions[i] = self.populations[i][idx]
        if self.best_fitnesses[i] < self.best_fitness_overall:
            self.best_fitness_overall = self.best_fitnesses[i]
            self.best_solution_overall = self.best_solutions[i]

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8, optimum_value=None) -> tuple:
        # ===== Initialization =====
        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.global_fitness_history.clear()
        self.last_global_restart_fe = -float('inf')

        # Per-island initialization
        for i in range(self.num_islands):
            self.fitness_values[i] = objective_function(self.populations[i])
            self.eval_count += self.population_size
            idx = np.argmin(self.fitness_values[i])
            self.best_fitnesses[i] = self.fitness_values[i][idx]
            self.best_solutions[i] = self.populations[i][idx]
            self.fitness_history[i].append(self.best_fitnesses[i])
            if self.best_fitnesses[i] < self.best_fitness_overall:
                self.best_fitness_overall = self.best_fitnesses[i]
                self.best_solution_overall = self.best_solutions[i]

        # ===== Main loop =====
        while self.eval_count < self.budget:
            if optimum_value is not None and abs(self.best_fitness_overall - optimum_value) <= acceptance_threshold:
                break

            # DE steps
            for i in range(self.num_islands):
                self.differential_evolution_step(i, objective_function)
                self.fitness_history[i].append(self.best_fitnesses[i])

            # Track global best
            self.global_fitness_history.append(self.best_fitness_overall)

            # Partial global restart (option A: reset subset)
            # -----------------------
            # Instead of full reset, we reinitialize only half the islands:
            # This preserves some existing good solutions to force a visible discontinuity.
            if (self.eval_count - self.last_global_restart_fe > 0.1 * self.budget
                    and self.detect_stagnation(self.global_fitness_history)):
                pre = self.best_fitness_overall
                logging.warning(f"[GLOBAL RESTART] before fitness={pre:.2e} at FE={self.eval_count}")
                islands = random.sample(range(self.num_islands), self.num_islands // 2)
                for j in islands:
                    self.populations[j] = np.random.uniform(
                        self.lower_bounds, self.upper_bounds,
                        (self.population_size, self.dim)
                    )
                    self.fitness_values[j] = objective_function(self.populations[j])
                    self.eval_count += self.population_size
                    idx = np.argmin(self.fitness_values[j])
                    self.best_fitnesses[j] = self.fitness_values[j][idx]
                    self.best_solutions[j] = self.populations[j][idx]
                best_idx = np.argmin(self.best_fitnesses)
                self.best_fitness_overall = self.best_fitnesses[best_idx]
                self.best_solution_overall = self.best_solutions[best_idx]
                post = self.best_fitness_overall
                logging.warning(f"[GLOBAL RESTART] after  fitness={post:.2e}")
                self.last_global_restart_fe = self.eval_count
                self.global_fitness_history.clear()

            # Partial global restart (option B: perturb around best)
            # -----------------------
            # Alternatively, you can sample all islands around the current best:
            # restart_scale = (np.linalg.norm(self.upper_bounds - self.lower_bounds) / 10)
            # center = self.best_solution_overall
            # for j in range(self.num_islands):
            #     noise = np.random.normal(0, restart_scale, (self.population_size, self.dim))
            #     self.populations[j] = np.clip(center + noise, self.lower_bounds, self.upper_bounds)
            #     self.fitness_values[j] = objective_function(self.populations[j])
            #     self.eval_count += self.population_size
            #     idx = np.argmin(self.fitness_values[j])
            #     self.best_fitnesses[j] = self.fitness_values[j][idx]
            #     self.best_solutions[j] = self.populations[j][idx]
            # self.best_fitness_overall = min(self.best_fitnesses)
            # self.last_global_restart_fe = self.eval_count
            # self.global_fitness_history.clear()

            # Migration
            if self.eval_count % self.migration_interval == 0:
                self.migrate(objective_function)

            # Local restarts
            for i in range(self.num_islands):
                if self.detect_stagnation(self.fitness_history[i]):
                    self.restart_island(i, objective_function)

        return self.best_solution_overall, self.best_fitness_overall, {'function_evaluations_used': self.eval_count}






def run_optimization(MaxEvals, AcceptanceThreshold, 
                     Dimension, CompNum, MinCoordinate, MaxCoordinate,
                     CompMinPos, CompSigma, CompH, Mu, Omega,
                     Lambda, RotationMatrix, OptimumValue, OptimumPosition,
                    num_runs: int = 5,
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
    best_values = []
    best_solutions = []
    aoccs = []
    for run in range(num_runs):
        run_seed = base_seed + run  # Vary seed per run
        np.random.seed(run_seed)
        random.seed(run_seed)
        gnbg = GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)
        bounds = (gnbg.MinCoordinate, gnbg.MaxCoordinate)
        logging.info(f"Starting run {run + 1}/{num_runs}")
        
        # Initialize algorithm
       
        try:
            optimizer = AdaptiveIslandDE(
                budget=MaxEvals,
                dim=gnbg.Dimension,
                lower_bounds=[gnbg.MinCoordinate for _ in range(gnbg.Dimension)],
                upper_bounds=[gnbg.MaxCoordinate for _ in range(gnbg.Dimension)],
            )
            
            # Run optimization
            best_solution, best_fitness, _ = optimizer.optimize(
                objective_function=gnbg.fitness, optimum_value = OptimumValue
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
    problem_list = [22]
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
        
        run_optimization(700000, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate,
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