import numpy as np
import random
class AdaptMigArchiveDE:
    """
    Adaptive Differential Evolution with Archive Bias, Migration, and Restarts. Enhanced for multimodal problems.
    """
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]):
        """
        Initializes the AdaptMigArchiveDE algorithm.

        Args:
            budget (int): The maximum number of function evaluations.
            dim (int): The dimensionality of the problem.
            lower_bounds (list[float]): The lower bounds of the search space.
            upper_bounds (list[float]): The upper bounds of the search space.
        """
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)

        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        self.population_size = 50
        self.num_sub_populations = 4

        self.populations = []
        for _ in range(self.num_sub_populations):
            population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
            self.populations.append(population)

        self.fitness_values = [np.full(self.population_size, float('inf')) for _ in range(self.num_sub_populations)]

        self.mutation_factor = 0.5
        self.crossover_rate = 0.7

        self.adaptive_f_memory = [[] for _ in range(self.num_sub_populations)]
        self.adaptive_cr_memory = [[] for _ in range(self.num_sub_populations)]

        self.archive_size = 10
        self.archives = [[] for _ in range(self.num_sub_populations)]  # Stores successful solutions
        self.archive_fitnesses = [[] for _ in range(self.num_sub_populations)] # Fitnesses of archived solutions

        self.num_generations_no_improvement = 0
        self.stagnation_threshold = 5000
        self.restart_probability = 0.05 # Chance to restart a subpopulation.
        self.p_selection = 0.8

        self.migration_interval = 5000 # Number of evals between migrations

        self.last_migration = 0 # Count evaluations since last migration


    def evaluate_population(self, population, objective_function: callable):
        """Evaluates the fitness of a population.

        Args:
            population (np.ndarray): The population to evaluate.
            objective_function (callable): The objective function.

        Returns:
            np.ndarray: The fitness values of the population.
        """
        fitness_values = objective_function(population)
        self.eval_count += len(population)
        return fitness_values

    def evolve_subpopulation(self, population_index: int, objective_function: callable):
        """Evolves a subpopulation using DE with archive bias and adaptive parameters.

        Args:
            population_index (int): The index of the subpopulation to evolve.
            objective_function (callable): The objective function.
        """
        population = self.populations[population_index]
        fitness = self.fitness_values[population_index]
        archive = self.archives[population_index]
        archive_fitness = self.archive_fitnesses[population_index]
        adaptive_f_memory = self.adaptive_f_memory[population_index]
        adaptive_cr_memory = self.adaptive_cr_memory[population_index]

        for i in range(self.population_size):
            # Selection
            if random.random() < self.p_selection:
                # Mutation: Using biased sampling from the archive
                if archive and random.random() < 0.3:
                    # Select an index from the archive according to fitness.

                    fitness_array = np.array(archive_fitness)
                    # Calculate Boltzmann probability if there are negative fitness values
                    if np.any(fitness_array < 0):
                        temperature = np.mean(np.abs(fitness_array))
                        probabilities = np.exp(-fitness_array / temperature)
                        probabilities /= np.sum(probabilities)
                        archive_index = np.random.choice(len(archive), p=probabilities)
                    else:
                        probabilities = 1 / (fitness_array + 1e-6)  # Avoid division by zero
                        probabilities /= np.sum(probabilities)
                        archive_index = np.random.choice(len(archive), p=probabilities)

                    biased_vector = archive[archive_index]
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b = random.sample(idxs, 2)
                    mutant = biased_vector + self.mutation_factor * (population[a] - population[b])

                else: # Regular DE mutation
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = random.sample(idxs, 3)
                    mutant = population[a] + self.mutation_factor * (population[b] - population[c])
            else:
                mutant = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)

            mutant = np.clip(mutant, self.lower_bounds, self.upper_bounds)

            # Crossover - Archive-biased crossover
            if archive and random.random() < 0.3:
                archive_index = random.randint(0, len(archive) - 1)
                archive_vector = archive[archive_index]
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant, archive_vector)
            else:
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant, population[i])

            # Evaluation
            trial_fitness = objective_function(trial_vector.reshape(1, -1))[0]
            self.eval_count += 1

            # Selection
            if trial_fitness < fitness[i]:
                # Update population
                population[i] = trial_vector
                fitness[i] = trial_fitness

                # Update archive (if applicable)
                if len(archive) < self.archive_size:
                    archive.append(trial_vector)
                    archive_fitness.append(trial_fitness)
                else:
                    # Replace the worst solution in the archive
                    max_archive_fitness_index = np.argmax(archive_fitness)
                    if trial_fitness < archive_fitness[max_archive_fitness_index]:
                        archive[max_archive_fitness_index] = trial_vector
                        archive_fitness[max_archive_fitness_index] = trial_fitness

                # Store successful F and CR
                adaptive_f_memory.append(self.mutation_factor)
                adaptive_cr_memory.append(self.crossover_rate)

                # Update the overall best solution and reset stagnation counter
                if trial_fitness < self.best_fitness_overall:
                    self.best_fitness_overall = trial_fitness
                    self.best_solution_overall = trial_vector
                    self.num_generations_no_improvement = 0
            else:
                self.num_generations_no_improvement += 1

        # Adaptive F and CR update
        if adaptive_f_memory:
            self.mutation_factor = np.median(adaptive_f_memory)
            self.crossover_rate = np.median(adaptive_cr_memory)
            adaptive_f_memory.clear()
            adaptive_cr_memory.clear()

        self.populations[population_index] = population
        self.fitness_values[population_index] = fitness
        self.archives[population_index] = archive
        self.archive_fitnesses[population_index] = archive_fitness
        self.adaptive_f_memory[population_index] = adaptive_f_memory
        self.adaptive_cr_memory[population_index] = adaptive_cr_memory

    def migrate(self):
        """Migrates individuals between subpopulations based on fitness."""
        # Sort subpopulations by best fitness
        sorted_indices = np.argsort([np.min(fitness) for fitness in self.fitness_values])
        worst_subpop_index = sorted_indices[-1]
        best_subpop_index = sorted_indices[0]

        # Replace worst individual in the worst subpopulation with a *random* individual from the best subpopulation
        worst_fitness_in_worst_subpop_index = np.argmax(self.fitness_values[worst_subpop_index])
        random_individual_in_best_subpop_index = random.randint(0, self.population_size - 1) # Select randomly instead of best

        self.populations[worst_subpop_index][worst_fitness_in_worst_subpop_index] = self.populations[best_subpop_index][random_individual_in_best_subpop_index].copy()
        self.fitness_values[worst_subpop_index][worst_fitness_in_worst_subpop_index] = self.fitness_values[best_subpop_index][random_individual_in_best_subpop_index]

    def reinitialize_population(self, population_index, objective_function: callable):
        """Re-initializes the population randomly with a diversity injection."""
        # Inject diversity by creating solutions far from the current best
        for i in range(self.population_size):
            self.populations[population_index][i] = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dim)
        self.fitness_values[population_index] = self.evaluate_population(self.populations[population_index], objective_function)
        best_index = np.argmin(self.fitness_values[population_index])
        if self.fitness_values[population_index][best_index] < self.best_fitness_overall:
            self.best_fitness_overall = self.fitness_values[population_index][best_index]
            self.best_solution_overall = self.populations[population_index][best_index]


    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8) -> tuple:
        """Optimizes the objective function using the AdaptMigArchiveDE algorithm.

        Args:
            objective_function (callable): The objective function to optimize.
            acceptance_threshold (float, optional): The acceptance threshold. Defaults to 1e-8.

        Returns:
            tuple: A tuple containing the best solution, best fitness, and optimization information.
        """
        self.eval_count = 0  # Reset for this run

        # Initialize populations
        for i in range(self.num_sub_populations):
            self.populations[i] = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
            self.fitness_values[i] = self.evaluate_population(self.populations[i], objective_function)

            best_index = np.argmin(self.fitness_values[i])
            if self.fitness_values[i][best_index] < self.best_fitness_overall:
                self.best_fitness_overall = self.fitness_values[i][best_index]
                self.best_solution_overall = self.populations[i][best_index]

        self.num_generations_no_improvement = 0
        self.last_migration = 0


        while self.eval_count < self.budget:
            for i in range(self.num_sub_populations):
                self.evolve_subpopulation(i, objective_function) #Evolve each subpopulation

            if self.eval_count - self.last_migration >= self.migration_interval:
                self.migrate()
                self.last_migration = self.eval_count

            #Check for stagnation and re-initialize if needed
            if self.num_generations_no_improvement > self.stagnation_threshold:
                subpop_to_reinitialize = random.randint(0, self.num_sub_populations - 1) #Random subpopulation
                self.reinitialize_population(subpop_to_reinitialize, objective_function)
                self.num_generations_no_improvement = 0


        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info