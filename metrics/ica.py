import numpy as np

def compute_ica_early_convergence_aware(fitness_history, optimum_value=None, acceptance_threshold=1e-1):
    if len(fitness_history) < 2:
        return 0.0

    fitness_history = np.array(fitness_history, dtype=np.float64)
    diffs = np.diff(fitness_history)
    improvements = np.clip(-diffs, 0, None)

    fitness_range = fitness_history.max() - fitness_history.min()
    if fitness_range == 0:
        return 1.0 if optimum_value is not None and abs(fitness_history[0] - optimum_value) <= acceptance_threshold else 0.0

    normalized_improvements = improvements / fitness_range

    # Assign weights: larger at the beginning, smaller toward the end
    weights = np.linspace(1.0, 0.0, num=len(normalized_improvements))

    weighted_sum = np.sum(normalized_improvements * weights)
    max_possible_weighted_sum = np.sum(weights)

    ica = weighted_sum / max_possible_weighted_sum

    # If convergence to optimum is reached early, give full score
    if optimum_value is not None:
        for i, val in enumerate(fitness_history):
            if abs(val - optimum_value) <= acceptance_threshold:
                early_score_boost = (len(fitness_history) - i) / len(fitness_history)
                return max(ica, early_score_boost)

    return float(ica)
