class AdaptiveIslandDE:
    def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float],
                 num_islands: int = 5, population_size: int = 20, crossover_rate: float = 0.8719572569354708,
                 mutation_rate: float = 0.6113964692124271, migration_interval: int = 896.9508697672186, migration_size: int = 2.414743986796276,
                 local_search_iterations: int = 5.957280686848644, local_search_perturbation_scale: float = 0.1446330223199665, restart_percentage: float = 0.7,stagnation_threshold: float = 1e-2, stagnation_patient: int = 200, adaptive_stagnation: bool = True, initial_stagnation_threshold: float = 1e-2, final_stagnation_threshold: float = 1e-8):
        """
        Initializes the AdaptiveIslandDE optimizer.

        Args:
            budget (int): Maximum number of function evaluations.
            dim (int): Problem dimensionality.
            lower_bounds (list[float]): List of lower bounds for each dimension.
            upper_bounds (list[float]): List of upper bounds for each dimension.
            num_islands (int): Number of independent subpopulations (islands).
            population_size (int): Number of individuals in each island's population.
            crossover_rate (float): DE crossover probability.
            mutation_rate (float): DE mutation scaling factor.
            migration_interval (int): Number of evaluations between migrations.
            migration_size (int): Number of individuals to migrate.
            local_search_iterations (int): Iterations for local search
            local_search_perturbation_scale (float): Scale of the random perturbation in local search.
            restart_percentage (float): Percentage of budget used to trigger restart.
        """
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
        self.restart_percentage = restart_percentage
        self.adaptive_stagnation = adaptive_stagnation
        # Restart parameters
        self.stagnation_patience = stagnation_patient
        self.stagnation_threshold = stagnation_threshold
        self.fitness_history = [[] for _ in range(num_islands)]
        
        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        # Initialize populations for each island
        self.populations = [
            np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
            for _ in range(self.num_islands)
        ]
        self.fitness_values = [np.full(self.population_size, float('inf')) for _ in range(self.num_islands)]
        self.best_solutions = [None] * self.num_islands
        self.best_fitnesses = [float('inf')] * self.num_islands


    def differential_evolution_step(self, island_index: int, objective_function: callable):
        """
        Performs a single step of differential evolution on a given island.

        Args:
            island_index (int): Index of the island to evolve.
            objective_function (callable): The objective function to optimize.
        """

        population = self.populations[island_index]
        fitness_values = self.fitness_values[island_index]

        for i in range(self.population_size):
            # Mutation
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = random.sample(indices, 3)
            mutant_vector = population[a] + self.mutation_rate * (population[b] - population[c])
            mutant_vector = np.clip(mutant_vector, self.lower_bounds, self.upper_bounds)  # Clip to bounds

            # Crossover
            trial_vector = np.copy(population[i])
            for j in range(self.dim):
                if random.random() < self.crossover_rate:
                    trial_vector[j] = mutant_vector[j]

            # Evaluation
            trial_vector_reshaped = trial_vector.reshape(1, -1)
            trial_fitness = objective_function(trial_vector_reshaped)[0]
            self.eval_count += 1

            # Selection
            if trial_fitness < fitness_values[i]:
                population[i] = trial_vector
                fitness_values[i] = trial_fitness

                # Update island best
                if trial_fitness < self.best_fitnesses[island_index]:
                    self.best_fitnesses[island_index] = trial_fitness
                    self.best_solutions[island_index] = trial_vector

                # Update overall best
                if trial_fitness < self.best_fitness_overall:
                    self.best_fitness_overall = trial_fitness
                    self.best_solution_overall = trial_vector

        self.populations[island_index] = population
        self.fitness_values[island_index] = fitness_values

    def local_search(self, solution: np.ndarray, objective_function: callable) -> tuple:
        """
        Performs local search around a solution using a simple gradient-based method.

        Args:
            solution (np.ndarray): The solution to start the local search from.
            objective_function (callable): The objective function to optimize.

        Returns:
            tuple: A tuple containing the improved solution and its fitness.
        """
        best_solution = solution.copy()
        best_fitness = objective_function(best_solution.reshape(1, -1))[0]
        self.eval_count += 1

        for _ in range(self.local_search_iterations):
            # Create a small random perturbation
            perturbation = np.random.normal(0, self.local_search_perturbation_scale, self.dim) # Scale adjusted from 1 to 0.1
            new_solution = best_solution + perturbation
            new_solution = np.clip(new_solution, self.lower_bounds, self.upper_bounds)

            # Evaluate the new solution
            new_fitness = objective_function(new_solution.reshape(1, -1))[0]
            self.eval_count += 1

            # If the new solution is better, update the current best
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_solution = new_solution

        return best_solution, best_fitness


    def migrate(self, objective_function: callable):
        """
        Migrates individuals between islands.  Each island sends its best
        individuals to a randomly chosen other island, and receives
        migrants to replace its worst individuals.
        """

        for i in range(self.num_islands):
            # Select a random destination island (excluding itself)
            dest_island = random.choice([j for j in range(self.num_islands) if j != i])

            # Identify the best solutions on the source island
            source_island_fitness = self.fitness_values[i]
            best_indices = np.argsort(source_island_fitness)[:self.migration_size]
            migrants = self.populations[i][best_indices].copy()  # Important to copy

            # Identify the worst solutions on the destination island
            dest_island_fitness = self.fitness_values[dest_island]
            worst_indices = np.argsort(dest_island_fitness)[-self.migration_size:]

            # Replace the worst solutions on the destination island with the migrants
            self.populations[dest_island][worst_indices] = migrants

            # Re-evaluate the fitness of the new solutions on the destination island (important!) and perform local search
            new_fitnesses = []
            for j in range(len(worst_indices)):
                migrant = migrants[j]
                migrant, fitness = self.local_search(migrant, objective_function) #Local Adaptation here.
                new_fitnesses.append(fitness)

            dest_island_fitness[worst_indices] = new_fitnesses
            self.fitness_values[dest_island] = dest_island_fitness
             # Update best fitness, if needed
            for fit, sol in zip(new_fitnesses, migrants):
                if fit < self.best_fitnesses[dest_island]:
                     self.best_fitnesses[dest_island] = fit
                     self.best_solutions[dest_island] = sol
                if fit < self.best_fitness_overall:
                    self.best_fitness_overall = fit
                    self.best_solution_overall = sol


    def detect_stagnation(self, fitness_history: list[float]) -> bool:
        if len(fitness_history) < self.stagnation_patience:
            return False
        recent = fitness_history[-self.stagnation_patience:]
    
        if self.adaptive_stagnation:
            # Log-scale interpolation
            log_start = np.log10(self.initial_stagnation_threshold)
            log_end = np.log10(self.final_stagnation_threshold)
            progress = self.eval_count / self.budget
            adaptive_threshold = 10 ** (log_start + progress * (log_end - log_start))
        else:
            adaptive_threshold = self.initial_stagnation_threshold

        std_recent = np.std(recent)
        
        # Optional debug
        logging.debug(f"FE={self.eval_count}, std={std_recent:.2e}, adaptive threshold={adaptive_threshold:.2e}")
        
        return std_recent < adaptive_threshold


    def restart_island(self, i: int, objective_function: callable):
        logging.info(f"Restarting island {i} due to stagnation at FE={self.eval_count}")
        self.populations[i] = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.population_size, self.dim))
        self.fitness_values[i] = objective_function(self.populations[i])
        self.eval_count += self.population_size
        best_index = np.argmin(self.fitness_values[i])
        self.best_fitnesses[i] = self.fitness_values[i][best_index]
        self.best_solutions[i] = self.populations[i][best_index]

        if self.best_fitnesses[i] < self.best_fitness_overall:
            self.best_fitness_overall = self.best_fitnesses[i]
            self.best_solution_overall = self.best_solutions[i]

    def optimize(self, objective_function: callable, acceptance_threshold: float = 1e-8, optimum_value=None) -> tuple:
        self.eval_count = 0
        self.best_solution_overall = None
        self.best_fitness_overall = float('inf')

        # Initialize fitness values for each island
        for i in range(self.num_islands):
            self.fitness_values[i] = objective_function(self.populations[i])
            self.eval_count += self.population_size
            best_index = np.argmin(self.fitness_values[i])
            self.best_fitnesses[i] = self.fitness_values[i][best_index]
            self.best_solutions[i] = self.populations[i][best_index]
            self.fitness_history[i].append(self.best_fitnesses[i])

            if self.best_fitnesses[i] < self.best_fitness_overall:
                self.best_fitness_overall = self.best_fitnesses[i]
                self.best_solution_overall = self.best_solutions[i]

        # Main optimization loop
        while self.eval_count < self.budget:
            if optimum_value is not None and abs(self.best_fitness_overall - optimum_value) <= acceptance_threshold:
                logging.info(f"Stopping early: Acceptance threshold {acceptance_threshold} reached at FE {self.eval_count}.")
                break

            for i in range(self.num_islands):
                self.differential_evolution_step(i, objective_function)
                self.fitness_history[i].append(self.best_fitnesses[i])

            if self.eval_count % self.migration_interval == 0:
                self.migrate(objective_function)

            # Restart islands based on stagnation detection
            for i in range(self.num_islands):
                if self.detect_stagnation(self.fitness_history[i]):
                    self.restart_island(i, objective_function)

        optimization_info = {
            'function_evaluations_used': self.eval_count,
            'final_best_fitness': self.best_fitness_overall,
        }
        return self.best_solution_overall, self.best_fitness_overall, optimization_info