from typing import Any, Dict, Hashable, List

from continuous_eval.eval.types import ToolCall
from continuous_eval.metrics.base import Field, Metric


def _count_matches(ground_truth, tools, order_sensitive=False):
    """
    Counts the number of matches between the ground truth and tools.
    For order-sensitive matching, it uses direct index comparison.
    For order-insensitive matching, it uses set intersections.
    """
    if order_sensitive:
        matches = 0
        gt_index = 0

        for tool in tools:
            if gt_index < len(ground_truth) and ground_truth[gt_index] == tool:
                matches += 1
                gt_index += 1
        return matches
    else:
        ground_truth_set = set(make_hashable(d) for d in ground_truth)
        tools_set = set(make_hashable(d) for d in tools)
        intersection = ground_truth_set & tools_set
        return len(intersection)


def make_hashable(obj: Any) -> Hashable:
    """
    Converts mutable types (list, dict) to immutable and hashable ones (tuple, frozenset).
    This helper function is used to facilitate hashable operations for dictionaries and lists.
    """
    if isinstance(obj, dict):
        return frozenset((k, make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, list):
        return tuple(make_hashable(v) for v in obj)
    else:
        return obj


class ToolSelectionAccuracy(Metric):
    """
    Computes the accuracy of tool selection.
    """

    def __init__(
        self,
        order_sensitive: bool = False,
        ignore_kwargs: bool = False,
    ) -> None:
        super().__init__(is_cpu_bound=True)
        self._order_sensitive = order_sensitive
        self._ignore_kwargs = ignore_kwargs

    def compute(
        self, tools: List[ToolCall], ground_truths: List[ToolCall], **kwargs
    ):
        if self._ignore_kwargs:
            _ground_truths = [{"name": t["name"]} for t in ground_truths]
            _tools = [{"name": t["name"]} for t in tools]
        else:
            _ground_truths, _tools = ground_truths, tools

        num_correct = _count_matches(
            _ground_truths, _tools, order_sensitive=self._order_sensitive
        )

        score = 1.0 if ground_truths else (0.0 if tools else 1.0)

        if ground_truths:
            score = num_correct / len(ground_truths)

        return {
            "num_correct": num_correct,
            "score": score,
        }

    @property
    def schema(self) -> Dict[str, Field]:
        return {
            "score": Field(type=float, limits=(0, 1)),
            "num_correct": Field(type=int),
        }
