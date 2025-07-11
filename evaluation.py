from scipy.io import loadmat
import os
import logging
from typing import Callable
import cma
from typing import Optional, Tuple, List
from metrics.ica import compute_ica_early_convergence_aware
from metrics.aocc import calculate_aocc_from_gnbg_history
import numpy as np
import cma
import random
from scipy.optimize import minimize
from gnbg.gnbg_python.GNBG_instances import GNBG
from datetime import datetime
import random
import matplotlib.pyplot as plt
folder = "logs/log_test_algorithms"
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

# Name: CMAESIslandAdaptiveDE
# Description: An island-based DE with adaptive migration, local search, and restart mechanisms to handle deceptive landscapes.
# Code:
# Terrible at: f14, f22
# Good at: f17, f18, f19, f13, f9, f20, f11, f12, f13

class ArchiveDE_SSA:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float],
                 pop_size: int = 1000, PD: float = 0.5, ST: float = 0.8, archive_size: int = 100):
        self.budget = budget
        self.dim = dim
        self.lower_bounds = np.array(lower_bounds, float)
        self.upper_bounds = np.array(upper_bounds, float)

        self.pop_size = pop_size
        self.PD = PD  # producer proportion
        self.ST = ST  # safety threshold
        self.archive_size = archive_size  # max archive capacity

        self.eval_count = 0
        self.best_solution = None
        self.best_fitness = float('inf')

        self.population = None
        self.fitness = None
        self.archive = []  # for archive-based DE

    def initialize_population(self):
        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds,
                                            (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

    def evaluate_population(self, objective_function):
        for i in range(self.pop_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = objective_function(self.population[i])
                self.eval_count += 1

        idx = np.argmin(self.fitness)
        if self.fitness[idx] < self.best_fitness:
            self.best_fitness = self.fitness[idx]
            self.best_solution = self.population[idx].copy()

    def clip(self, x):
        return np.clip(x, self.lower_bounds, self.upper_bounds)

    def differential_evolution_mutation(self, i):
        # Use archive + current population for DE mutation
        candidates = list(range(self.pop_size))
        candidates.remove(i)
        r1, r2 = random.sample(candidates, 2)

        if self.archive:
            r3 = random.randrange(len(self.archive))
            x_r3 = self.archive[r3]
        else:
            r3 = r2
            x_r3 = self.population[r3]

        F = np.random.uniform(0.7, 0.9)
        mutant = self.population[i] + F * (self.population[r1] - x_r3)
        return self.clip(mutant)

    def optimize(self, objective_function: Callable[[np.ndarray], float]):
        self.init = True
        self.initialize_population()
        self.evaluate_population(objective_function)

        while self.eval_count < self.budget:
            # 1️⃣ SSA Producers & Scroungers
            n_prod = int(self.PD * self.pop_size)
            worst_idx = np.argmax(self.fitness)
            best_idx = np.argmin(self.fitness)
            x_best = self.population[best_idx]
            x_worst = self.population[worst_idx]

            for i in range(self.pop_size):
                if i < n_prod:
                    # Producer
                    r = np.random.rand()
                    if r < self.ST:
                        self.population[i] *= np.exp(-i / (np.random.rand() * 1e-3 + 1e-3))
                    else:
                        self.population[i] += np.random.randn(self.dim)
                else:
                    # Scrounger
                    if i > self.pop_size / 2:
                        self.population[i] += np.random.randn(self.dim) * np.abs(self.population[i] - x_worst)
                    else:
                        self.population[i] += np.abs(self.population[i] - x_best) * np.random.randn(self.dim)

                self.population[i] = self.clip(self.population[i])

            # 2️⃣ Archive-based DE variation
            for i in range(self.pop_size):
                mutant = self.differential_evolution_mutation(i)
                trial = np.copy(self.population[i])
                CR = np.random.uniform(0.6, 0.9)
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == jrand:
                        trial[j] = mutant[j]
                trial = self.clip(trial)

                f_trial = objective_function(trial)
                self.eval_count += 1

                # Select & store into archive
                if f_trial < self.fitness[i]:
                    # update archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(self.population[i].copy())
                    else:
                        # Replace worst in archive
                        # randomly drop one
                        idx = np.random.randint(self.archive_size)
                        self.archive[idx] = self.population[i].copy()

                    self.population[i] = trial
                    self.fitness[i] = f_trial

                    if f_trial < self.best_fitness:
                        self.best_fitness = f_trial
                        self.best_solution = trial.copy()

                if self.eval_count >= self.budget:
                    break

            # 3️⃣ Vigilance-awareness update
            n_vig = max(1, int(0.1 * self.pop_size))
            v_idx = random.sample(range(self.pop_size), n_vig)
            for idx in v_idx:
                if self.fitness[idx] > self.best_fitness:
                    self.population[idx] = x_best + np.random.rand(self.dim) * np.abs(self.population[idx] - x_best)
                else:
                    self.population[idx] += np.random.randn(self.dim) * np.abs(self.population[idx] - x_worst)
                self.population[idx] = self.clip(self.population[idx])

            self.evaluate_population(objective_function)

        return self.best_solution, self.best_fitness, {'function_evaluations_used': self.eval_count}

def run_optimization(MaxEvals, AcceptanceThreshold, 
                     Dimension, CompNum, MinCoordinate, MaxCoordinate,
                     CompMinPos, CompSigma, CompH, Mu, Omega,
                     Lambda, RotationMatrix, OptimumValue, OptimumPosition,
                    num_runs: int = 10,
                    seed: Optional[int] = None) -> Tuple[List[float], List[np.ndarray]]:
  
    base_seed = 42
    
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
                        
            try:
                optimizer = ArchiveDE_SSA(
                    budget=MaxEvals,
                    dim=gnbg.Dimension,
                    lower_bounds=[gnbg.MinCoordinate for _ in range(gnbg.Dimension)],
                    upper_bounds=[gnbg.MaxCoordinate for _ in range(gnbg.Dimension)]
                )
                
                # Run optimization
                best_solution, best_fitness, _ = optimizer.optimize(
                    objective_function=gnbg.fitness)
                best_values.append(best_fitness)
                best_params.append(best_solution)
                
                aocc = calculate_aocc_from_gnbg_history(
                    fe_history=gnbg.FEhistory,
                    optimum_value=gnbg.OptimumValue,
                    budget_B=gnbg.MaxEvals
                )
                ica = compute_ica_early_convergence_aware(fitness_history=gnbg.FEhistory, optimum_value=gnbg.OptimumValue)
                best_solutions.append(best_solution)
                aoccs.append(aocc)
                
                logging.info(f"Run {run + 1} completed. Best fitness: {best_fitness}, AOCC: {aocc}")
                logging.info(f"\nResults for Problem {ProblemIndex}:")
                logging.info(f"History: {gnbg.FEhistory}")
                logging.info(f"Best solution: {best_solutions}")
                logging.info(f"Optimun Solution: {OptimumValue}")
                logging.info(f"Best fitness values: {best_values}")
                logging.info(f"Mean fitness: {np.mean(best_values)}")
                logging.info(f"Std fitness: {np.std(best_values)}") 
                logging.info(f"Mean AOCC:         {np.mean(aoccs):.4f} (Higher is better)")
                logging.info(f"ICA:         {ica} (Higher is better)")
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