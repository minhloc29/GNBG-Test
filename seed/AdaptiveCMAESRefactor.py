import numpy as np
import random
import cma
import logging

# Name: AdaptiveIslandCMAES_Refactored
# Description: Uses an island model with CMA-ES and adapts exploration based on island performance and restarts.
# Code:
class AdaptiveIslandCMAES_Refactored:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float],
                 num_islands: int = 5, initial_population_size: int = 10, initial_sigma: float = 1.0,
                 max_restarts: int = 3, population_size_multiplier: float = 2.0,
                 migration_interval: int = 1000, migration_size: int = 2, adaptation_rate: float = 0.1744806071029344,
                 success_threshold: float = 0.19363661844259009, sigma_reduction_factor: float = 0.578982538692469,
                 sigma_increase_factor: float = 1.6777965479936499, sigma_clip_factor: float = 6.829918880113372,
                 max_population_size_factor: int = 127.28789619324785, migration_fitness_improvement_threshold: float = 0.020719790478888722):
        """
        Initializes the AdaptiveIslandCMAES optimizer.

        Args:
            budget (int): Maximum number of function evaluations.
            dim (int): Problem dimensionality.
            lower_bounds (list[float]): List of lower bounds for each dimension.
            upper_bounds (list[float]): List of upper bounds for each dimension.
            num_islands (int): Number of islands in the island model.
            initial_population_size (int): Initial population size for each island.
            initial_sigma (float): Initial standard deviation for search.
            max_restarts (int): Maximum number of restarts allowed for each island.
            population_size_multiplier (float): Factor by which population size increases on restart.
            migration_interval (int): Number of evaluations between migrations.
            migration_size (int): Number of individuals to migrate.
            adaptation_rate (float): Rate at which sigma is adjusted based on island performance.
            success_threshold (float): Threshold for considering a run successful.
            sigma_reduction_factor (float): Factor to reduce sigma when a run is successful.
            sigma_increase_factor (float): Factor to increase sigma when a run is unsuccessful.
            sigma_clip_factor (float): Factor to clip sigma within a reasonable range.
            max_population_size_factor (int): Factor to limit the maximum population size.
            migration_fitness_improvement_threshold (float): Minimum relative improvement required for migration.
        """
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)
        self.num_islands = num_islands
        self.initial_population_size = initial_population_size
        self.initial_sigma = initial_sigma
        self.max_restarts = max_restarts
        self.population_size_multiplier = population_size_multiplier
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.adaptation_rate = adaptation_rate
        self.success_threshold = success_threshold
        self.sigma_reduction_factor = sigma_reduction_factor
        self.sigma_increase_factor = sigma_increase_factor
        self.sigma_clip_factor = sigma_clip_factor
        self.max_population_size_factor = max_population_size_factor
        self.migration_fitness_improvement_threshold = migration_fitness_improvement_threshold

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')
        self.islands = []
        for _ in range(num_islands):
            self.islands.append({
                'best_solution': None,
                'best_fitness': float('inf'),
                'cma_es': None,
                'population_size': self.initial_population_size,
                'sigma': self.initial_sigma,
                'success_rate': 0.0
            })
        self.last_migration_eval = 0

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8, optimum_value=None) -> tuple:
        """
        Optimizes the given objective function using an adaptive island model with CMA-ES.

        Args:
            objective_function (callable): The objective function to optimize.
            acceptance_threshold (float): Threshold for accepting a solution as the optimum.
            optimum_value (float, optional): Known optimum value for early stopping. Defaults to None.

        Returns:
            tuple: A tuple containing the best solution, its fitness, and optimization information.
        """
        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        while self.eval_count < self.budget:
            # Island-specific optimization
            for i in range(self.num_islands):
                if self.eval_count >= self.budget:
                    break
                island = self.islands[i]
                num_restarts = 0
                while self.eval_count < self.budget and num_restarts <= self.max_restarts:

                    logging.info(f"Island {i+1}: Starting new run (restart {num_restarts + 1}) with pop size {island['population_size']} at FE={self.eval_count}, sigma={island['sigma']:.3f}")

                    x0 = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim) # Island restarts start from random points.

                    options = {
                        'bounds': [list(self.lower_bounds), list(self.upper_bounds)],
                        'popsize': island['population_size'],
                        'maxfevals': self.budget - self.eval_count, # Remaining budget
                        'verb_disp': 0, # No status messages
                        'seed': np.random.randint(0, 1000000), # Random seed for each restart
                    }

                    def cma_objective_wrapper(x):
                        if self.eval_count >= self.budget:
                            return float('inf')

                        fitness_val = objective_function(x.reshape(1, -1))[0]
                        self.eval_count += 1

                        if fitness_val < island['best_fitness']:
                            island['best_fitness'] = fitness_val
                            island['best_solution'] = x.copy()
                            island['success_rate'] = 1.0 # Mark as successful run

                        if fitness_val < self.best_fitness_overall:
                            self.best_fitness_overall = fitness_val
                            self.best_solution_overall = x.copy()
                            logging.info(f"New global best: {self.best_fitness_overall:.6e} at FE={self.eval_count}")


                        return fitness_val


                    try:
                        result = cma.fmin(cma_objective_wrapper, x0, island['sigma'], options)
                        result_solution = result[0]
                        result_fitness = result[1]
                        evaluations_used_by_cma = result[2] # result[2] has all the evaluations used in the cmaes run.

                        # Update overall bests again in case the loop terminated but a better solution was found
                        if result_fitness < self.best_fitness_overall:
                            self.best_fitness_overall = result_fitness
                            self.best_solution_overall = result_solution

                        if self.eval_count >= self.budget or (optimum_value is not None and abs(self.best_fitness_overall - optimum_value) <= acceptance_threshold):
                            break # Terminate if overall budget reached or optimum found

                        logging.info(f"Island {i+1}: CMA-ES run completed. Best fitness: {result_fitness:.6e}, FE used: {evaluations_used_by_cma}")

                    except Exception as e:
                        logging.error(f"Island {i+1}: CMA-ES run failed due to: {e}", exc_info=True)
                        if self.eval_count >= self.budget:
                            break

                    # Sigma adaptation
                    if island['success_rate'] > self.success_threshold:  # Run was "successful" (found a new best)
                        island['sigma'] *= (1 - self.adaptation_rate) * self.sigma_reduction_factor # Reduce sigma, focus search
                    else:
                        island['sigma'] *= (1 + self.adaptation_rate) * self.sigma_increase_factor # Increase sigma, explore more
                    island['success_rate'] = 0.0 # Reset success rate
                    island['sigma'] = np.clip(island['sigma'], self.initial_sigma/self.sigma_clip_factor, self.initial_sigma*self.sigma_clip_factor) # keep sigma in a reasonable range.

                    num_restarts += 1
                    island['population_size'] = int(island['population_size'] * self.population_size_multiplier)

                    if island['population_size'] > self.max_population_size_factor * self.dim:
                        logging.warning(f"Island {i+1}: Population size grew very large ({island['population_size']}). Capping for next restart.")
                        island['population_size'] = self.max_population_size_factor * self.dim

            # Migration
            if self.eval_count - self.last_migration_eval >= self.migration_interval and self.eval_count < self.budget:
                self.migrate()
                self.last_migration_eval = self.eval_count

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall,
            'num_islands': self.num_islands
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info

    def migrate(self):
        """
        Migrates individuals between islands.  Worse islands send individuals to better islands.
        """
        logging.info("Starting migration...")
        # Sort islands by fitness (ascending order)
        sorted_islands = sorted(range(self.num_islands), key=lambda i: self.islands[i]['best_fitness'])

        # Migrate from worse to better islands
        for i in range(self.num_islands // 2): # Only migrate from the worse half
            source_island_index = sorted_islands[self.num_islands - 1 - i] # Worse islands
            destination_island_index = sorted_islands[i] # Better islands

            source_island = self.islands[source_island_index]
            dest_island = self.islands[destination_island_index]

            # Select migrants (best from source island)
            migrants = source_island['best_solution']
            if migrants is None:
                logging.info(f"Island {source_island_index} has no solution to migrate.")
                continue

            # Only migrate if source island fitness better *or* destination island has no solution.
            fitness_improvement = (dest_island['best_fitness'] - source_island['best_fitness']) / abs(dest_island['best_fitness']) if dest_island['best_fitness'] != 0 else float('inf')
            if source_island['best_fitness'] < dest_island['best_fitness'] or dest_island['best_solution'] is None or fitness_improvement > self.migration_fitness_improvement_threshold:
                dest_island['best_solution'] = migrants.copy()
                dest_island['best_fitness'] = source_island['best_fitness'] # Update fitness with *source* fitness
                logging.info(f"Migration: Island {source_island_index} -> Island {destination_island_index}, fitness {source_island['best_fitness']:.6e}")
            else:
                logging.info(f"Migration skipped: Island {source_island_index} -> Island {destination_island_index}, source fitness not better.")

            # Update overall best if needed, using the destination island as the source if the destination island has better fitness, or the source if its fitness is better.
            if dest_island['best_fitness'] < self.best_fitness_overall:
                  self.best_fitness_overall = dest_island['best_fitness']
                  self.best_solution_overall = dest_island['best_solution'].copy()
            if source_island['best_fitness'] < self.best_fitness_overall:
                  self.best_fitness_overall = source_island['best_fitness']
                  self.best_solution_overall = source_island['best_solution'].copy()

