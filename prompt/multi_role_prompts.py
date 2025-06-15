role1 = '''
Your objective is to design a novel optimization algorithm for a specific class of GNBG problems (like f8). These functions have a single global basin of attraction, but their interior is highly rugged, filled with a very large number of narrow local optima.
The primary challenge is to escape these numerous traps. An algorithm with a fixed step size will either be too slow to move or will jump over the global optimum. Your task is to provide an excellent and novel algorithm that uses an adaptive mutation strategy. The mutation strength should be large when the algorithm is stagnating (stuck in a local optimum) and should become smaller when it is making consistent progress (exploiting a promising region).
'''

role2 = '''
Your objective is to design a novel optimization algorithm for a specific class of GNBG problems (like f8). These functions have a single global basin of attraction, but their interior is highly rugged, filled with a very large number of narrow local optima.
The primary challenge is to effectively search this "spiky" landscape. Your task is to provide an excellent and novel algorithm that hybridizes a global search with a powerful local search. The algorithm should use a global method (like a simplified PSO or GA) to broadly navigate between regions of local optima, and then periodically trigger a dedicated local search method (like Nelder-Mead or Powell's method from scipy.optimize) on the best-so-far solution to precisely find the bottom of the current local "dip".
'''
role3 = '''
Your objective is to design a novel optimization algorithm for a specific class of GNBG problems (like f8). These functions have a single global basin of attraction, but their interior is highly rugged, filled with a very large number of narrow local optima.
The key challenge is to distinguish between the many local optima to find the global one. Your task is to provide an excellent and novel algorithm that uses fitness landscape topography to guide its search. Instead of just using the fitness value, the algorithm should consider the relationship between neighboring solutions. For example, it could:
Implement a "clearing" or "derating" procedure where solutions that are too close to an already discovered (and better) local optimum have their fitness penalized, encouraging the search to move elsewhere.
Use the fitness values of a small neighborhood of points around a candidate solution to estimate a local gradient or curvature, using this information to influence the next search step.
'''

role4= '''
Your objective is to design a novel optimization algorithm for a specific class of GNBG problems (like f8). These functions have a single global basin of attraction, but their interior is highly rugged, filled with a very large number of narrow local optima.

The challenge is to maintain exploration while exploiting good solutions. Your task is to provide an excellent and novel algorithm inspired by archive-guided Differential Evolution (like JADE or SHADE). The algorithm should maintain an external archive of recently successful solutions. The DE mutation strategy (e.g., DE/current-to-pbest/1) should select one of the "best" parents (pbest) from this archive, rather than just from the current population, to guide the generation of new candidate solutions.'''

role5 = '''
Your objective is to design a novel optimization algorithm for a specific class of GNBG problems (like f8). These functions have a single global basin of attraction, but their interior is highly rugged, filled with a very large number of narrow local optima.

The challenge is to adapt the search behavior dynamically. Your task is to provide an excellent and novel algorithm that uses an ensemble of different search strategies. In each generation, for each individual, the algorithm should probabilistically choose one of several search operators to apply. For example, the operators could be:

A standard DE mutation (for exploitation).
A strong "exploratory" Gaussian mutation with a large step size.
A simple local search move.

The probabilities of choosing each operator could be adaptive based on which operators have led to improvements recently.
'''