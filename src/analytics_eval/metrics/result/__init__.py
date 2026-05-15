"""Result quality metrics — evaluate query output correctness.

Available metrics:
- ResultAccuracy: Numerical comparison with configurable tolerance
- GrainCorrectness: Check aggregation level of results
"""

from analytics_eval.metrics.result.grain_correctness import GrainCorrectness
from analytics_eval.metrics.result.result_accuracy import ResultAccuracy

__all__ = ["ResultAccuracy", "GrainCorrectness"]
