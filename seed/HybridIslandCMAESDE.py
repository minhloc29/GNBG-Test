import numpy as np
import random
import cma

# Name: HybridIslandCMAESDE
# Description: Combines CMA-ES islands with adaptive sigma, dynamic DE migration, and a diversity maintenance strategy.
# Code:
class HybridIslandCMAESDE:
    """
    A hybrid optimization algorithm combining CMA-ES, Differential Evolution (DE),
    and an island model with adaptive sigma and dynamic migration for enhanced
    exploration and exploitation in multi-component landscapes.
    """
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float],
                 num_islands: int = 5, initial_population_size: int = 390, initial_sigma: float = 7.035340028919247,
                 max_restarts: int = 30, de_mutation_factor: float = 0.84, de_crossover_rate: float = 0.8534144807510109,
                 adaptation_rate: float = 0.48, success_threshold: float = 0.11778146580610886,
                 sigma_reduction_factor: float = 0.19975419579735468, sigma_increase_factor: float = 4,
                 sigma_clip_factor: float = 9.48777989161599, de_local_search_prob: float = 0.98,
                 migration_interval: int = 660.6234424876376, migration_size: int = 3, diversity_threshold: float = 0.07375074133172785,
                 de_local_search_popsize: int = 70, de_local_search_iterations: int = 50,
                 exploration_increase_factor: float = 5.84):
        """
        Initializes the HybridIslandCMAESDE optimizer.

        Args:
            budget (int): Maximum number of function evaluations.
            dim (int): Problem dimensionality.
            lower_bounds (list[float]): List of lower bounds for each dimension.
            upper_bounds (list[float]): List of upper bounds for each dimension.
            num_islands (int): Number of islands in the island model.
            initial_population_size (int): Initial CMA-ES population size for each island.
            initial_sigma (float): Initial CMA-ES standard deviation for search.
            max_restarts (int): Maximum CMA-ES restarts allowed for each island.
            de_mutation_factor (float): DE mutation factor.
            de_crossover_rate (float): DE crossover rate.
            adaptation_rate (float): Rate at which CMA-ES sigma is adjusted based on island performance.
            success_threshold (float): Threshold for considering a CMA-ES run successful.
            sigma_reduction_factor (float): Factor to reduce CMA-ES sigma when a run is successful.
            sigma_increase_factor (float): Factor to increase CMA-ES sigma when a run is unsuccessful.
            sigma_clip_factor (float): Factor to clip CMA-ES sigma within a reasonable range.
            de_local_search_prob (float): Probability of performing local DE search.
            migration_interval (int): Number of function evaluations between migrations.
            migration_size (int): Number of individuals to migrate.
            diversity_threshold (float): Threshold for population diversity; triggers enhanced exploration.
            de_local_search_popsize (int): Population size for local DE search.
            de_local_search_iterations (int): Number of iterations for local DE search.
            exploration_increase_factor (float): Factor to increase sigma during exploration enhancement.
        """
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)
        self.num_islands = num_islands
        self.initial_population_size = initial_population_size
        self.initial_sigma = initial_sigma
        self.max_restarts = max_restarts
        self.de_mutation_factor = de_mutation_factor
        self.de_crossover_rate = de_crossover_rate
        self.adaptation_rate = adaptation_rate
        self.success_threshold = success_threshold
        self.sigma_reduction_factor = sigma_reduction_factor
        self.sigma_increase_factor = sigma_increase_factor
        self.sigma_clip_factor = sigma_clip_factor
        self.de_local_search_prob = de_local_search_prob
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.diversity_threshold = diversity_threshold
        self.de_local_search_popsize = de_local_search_popsize
        self.de_local_search_iterations = de_local_search_iterations
        self.exploration_increase_factor = exploration_increase_factor

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
        self.migration_counter = 0

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8, optimum_value = None) -> tuple:
        """
        Optimizes the given objective function using CMA-ES with adaptive sigma on islands,
        plus adaptively randomized DE for local refinement and dynamic island migration.

        Args:
            objective_function (callable): The objective function to optimize.
            acceptance_threshold (float): Threshold for accepting a solution as the optimum.

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

                    # CMA-ES Phase
                    x0 = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
                    options = {
                        'bounds': [list(self.lower_bounds), list(self.upper_bounds)],
                        'popsize': island['population_size'],
                        'maxfevals': self.budget - self.eval_count,
                        'verb_disp': 0
                    }

                    def cma_objective_wrapper(x):
                        if self.eval_count >= self.budget:
                            return float('inf')

                        fitness_val = objective_function(x.reshape(1, -1))[0]
                        self.eval_count += 1

                        if fitness_val < island['best_fitness']:
                            island['best_fitness'] = fitness_val
                            island['best_solution'] = x.copy()
                            island['success_rate'] = 1.0

                        if fitness_val < self.best_fitness_overall:
                            self.best_fitness_overall = fitness_val
                            self.best_solution_overall = x.copy()

                        return fitness_val

                    try:
                        result = cma.fmin(cma_objective_wrapper, x0, island['sigma'], options)
                    except Exception as e:
                        if self.eval_count >= self.budget:
                            break
                        num_restarts += 1
                        continue

                    # Sigma adaptation
                    if island['success_rate'] > self.success_threshold:
                        island['sigma'] *= self.sigma_reduction_factor
                    else:
                        island['sigma'] *= self.sigma_increase_factor
                    island['success_rate'] = 0.0
                    island['sigma'] = np.clip(island['sigma'], self.initial_sigma/self.sigma_clip_factor, self.initial_sigma*self.sigma_clip_factor)

                    # Adaptive DE Local Search around best CMA-ES solution
                    if random.random() < self.de_local_search_prob and island['best_solution'] is not None:
                        self.adaptive_de_local_search(objective_function, island)

                    num_restarts += 1

            # Dynamic Island Migration
            if self.eval_count - self.migration_counter >= self.migration_interval:
                self.migrate_individuals(objective_function)
                self.migration_counter = self.eval_count

            # Diversity Check
            if self.check_diversity() < self.diversity_threshold:
                self.enhance_exploration()

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall,
            'num_islands': self.num_islands
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info

    def adaptive_de_local_search(self, objective_function: callable, island: dict):
        """
        Performs an adaptive differential evolution local search around the island's best solution.
        """
        popsize = self.de_local_search_popsize # keep population size small for local search
        population = np.random.normal(island['best_solution'], island['sigma'], size=(popsize, self.dim))
        population = np.clip(population, self.lower_bounds, self.upper_bounds)
        fitness = np.array([objective_function(x.reshape(1, -1))[0] for x in population])
        self.eval_count += popsize
        best_index = np.argmin(fitness)
        best_fitness = fitness[best_index]
        best_solution = population[best_index].copy()

        # DE iterations
        for _ in range(self.de_local_search_iterations): # small number of iterations
            for i in range(popsize):
                idxs = [idx for idx in range(popsize) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.de_mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds)

                trial = np.copy(population[i])
                for k in range(self.dim):
                    if random.random() < self.de_crossover_rate:
                        trial[k] = mutant[k]
                trial = np.clip(trial, self.lower_bounds, self.upper_bounds)

                trial_fitness = objective_function(trial.reshape(1, -1))[0]
                self.eval_count += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial.copy()
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial.copy()

        if best_fitness < island['best_fitness']:
            island['best_fitness'] = best_fitness
            island['best_solution'] = best_solution.copy()
            if best_fitness < self.best_fitness_overall:
                self.best_fitness_overall = best_fitness
                self.best_solution_overall = best_solution.copy()

    def migrate_individuals(self, objective_function: callable):
        """
        Migrates individuals between islands to promote diversity.
        """
        # Select the best individuals from each island
        best_individuals = []
        for island in self.islands:
            best_individuals.append((island['best_solution'], island['best_fitness']))

        # Sort the islands based on their best fitness
        sorted_islands = sorted(range(len(self.islands)), key=lambda k: self.islands[k]['best_fitness'])

        # Migrate individuals from better islands to worse islands
        num_to_migrate = min(self.migration_size, len(self.islands))
        for i in range(num_to_migrate):
            source_island_index = sorted_islands[i]
            destination_island_index = sorted_islands[-(i + 1)]  # Migrate to the worst islands

            # Replace the worst individuals on the destination island with the best individuals from the source island
            self.islands[destination_island_index]['best_solution'] = best_individuals[source_island_index][0]
            self.islands[destination_island_index]['best_fitness'] = best_individuals[source_island_index][1]


    def check_diversity(self):
        """
        Checks the diversity of the islands' best solutions.
        """
        solutions = [island['best_solution'] for island in self.islands if island['best_solution'] is not None]

        if not solutions:
             return 1.0  # High diversity if there are no solutions yet

        solutions = np.array(solutions)
        centroid = np.mean(solutions, axis=0)
        distances = np.linalg.norm(solutions - centroid, axis=1) # Euclidean distances
        diversity = np.std(distances) # Standard deviation of distances from the centroid

        # Normalize the diversity by the range of the search space
        diversity /= np.linalg.norm(self.upper_bounds - self.lower_bounds)

        return diversity


    def enhance_exploration(self):
        """
        Increases exploration by re-initializing islands with low diversity.
        """
        for island in self.islands:
            # Increase sigma
            island['sigma'] *= self.sigma_increase_factor*self.exploration_increase_factor
            island['sigma'] = np.clip(island['sigma'], self.initial_sigma/self.sigma_clip_factor, self.initial_sigma*self.sigma_clip_factor)

            # Re-initialize the CMA-ES population
            island['best_solution'] = None
            island['best_fitness'] = float('inf')