# f9
role1 = ''' 
**(CR: Capacity and Role)**
Act as an expert in designing advanced metaheuristic algorithms that can solve deceptive, high-dimensional optimization problems. Your expertise is in creating strategies that can reliably escape deep local optima.

**(I: Insight)**
Your objective is to design a **novel optimization algorithm** in Python to solve the GNBG `f9` benchmark problem. The defining characteristic of `f9` is that its landscape is filled with **numerous, significantly deep local optima** (due to high `Mu` parameters).

The primary challenge is that any standard greedy or evolutionary algorithm will quickly fall into one of these deep traps and get stuck permanently, resulting in poor performance. A better initialization is not enough; the algorithm's **core search dynamics must include an explicit mechanism to escape a deep basin of attraction** even after it has converged.

**(S: Statement)**
Provide a complete, standalone Python class for the optimization algorithm. The class **must** adhere to the following signature:

`def __init__(self, budget: int, dim: int, lower_bounds: list[float], upper_bounds: list[float]) -> None:`
* `n_pop`: The desired number of individuals in the population (population size).
* `dim`: The dimensionality of the problem (typically 30).
* `lower_bounds`, `upper_bounds`: Lists of floats for the variable boundaries (typically -100.0 to 100.0).
* The function must return a **2D NumPy array** of shape `(n_pop, dim)`, where each row is a solution vector and each value is within the specified bounds.

**(P: Personality)**
Provide an **excellent and sophisticated** algorithm that directly addresses the challenge of escaping deep local optima. Do not simply return a standard GA or DE. Instead, implement a **novel escape mechanism**. Inspiration can come from:
* **Simulated Annealing Principles:** Incorporate a "temperature" parameter and an acceptance criterion that allows the algorithm to probabilistically accept worse moves, with the probability decreasing over time.
* **Tabu Search Principles:** Maintain a short-term memory (a "tabu list") of recently visited local optima. The algorithm should be penalized or forbidden from returning to these tabu regions, forcing it to explore new areas.
* **Iterated Local Search (ILS):** Design a two-phase loop. First, use a strong local search to find the bottom of a basin. Second, apply a powerful "perturbation" or "kick" to the solution to jump into a new basin of attraction, and then restart the local search from there.
* **Fitness Derating/Sharing:** Implement a mechanism where the fitness of individuals in a crowded region is penalized, giving individuals in less-explored valleys a chance to reproduce.
'''

role2 = '''
Your objective is to design a novel optimization algorithm for GNBG problems (like f9) that feature a single basin of attraction filled with numerous, very deep local optima. The key challenge is to prevent the search from repeatedly re-visiting the same deep valleys.
Your task is to provide an excellent and novel algorithm inspired by Tabu Search. The algorithm should maintain a short-term memory (a "tabu list") of recently visited solutions or regions. New candidate solutions that are too close to a solution on the tabu list should be penalized or discarded, forcing the search to move to unexplored areas and preventing it from getting stuck in the deepest local optimum it has found so far.
'''
role3 = '''
Your objective is to design a novel optimization algorithm for GNBG problems (like f9) that feature a single basin of attraction filled with numerous, very deep local optima. The key challenge is to prevent the search from repeatedly re-visiting the same deep valleys.
'''

role4= '''
Your objective is to design a novel optimization algorithm for GNBG problems (like f9) that feature a single basin of attraction filled with numerous, very deep local optima. The challenge is to adapt the search strategy to either explore or exploit as needed.
Your task is to provide an excellent and novel algorithm based on Differential Evolution that uses an ensemble of mutation strategies. The algorithm should maintain a pool of DE strategies (e.g., DE/rand/1, DE/best/1, DE/current-to-pbest/1). In each generation, it should probabilistically choose which strategy to use for creating new individuals, and it should adapt these probabilities based on which strategies have been most successful at generating improvements in recent generations.
'''
role5 = '''
Your objective is to design a novel optimization algorithm for GNBG problems (like f9) that feature a single basin of attraction filled with numerous, very deep local optima. A simple local search will get stuck, but a purely global search may not be efficient.
Your task is to provide an excellent and novel algorithm based on Iterated Local Search (ILS). The algorithm's core loop should consist of two phases:
A Local Search phase that greedily finds the bottom of the current basin (a local optimum).
A Perturbation phase that applies a strong, random "kick" or mutation to the found local optimum to escape its basin of attraction and start a new local search in a different region of the landscape.
'''