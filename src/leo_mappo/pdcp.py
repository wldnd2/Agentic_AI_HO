from __future__ import annotations

import numpy as np


def joint_delivery_probability(primary_success: float, secondary_success: float, failure_correlation: float = 0.0) -> float:
    """Approximate probability that at least one duplicated leg succeeds.

    correlation=0 gives conditionally independent failures. Positive correlation
    reduces diversity gain without changing either marginal probability.
    """
    p1 = float(np.clip(primary_success, 0.0, 1.0))
    p2 = float(np.clip(secondary_success, 0.0, 1.0))
    independent = 1.0 - (1.0 - p1) * (1.0 - p2)
    no_diversity = max(p1, p2)
    return float((1.0 - failure_correlation) * independent + failure_correlation * no_diversity)

