from typing import Any, Dict, List, Literal, Set, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from continuous_eval.metrics.base import Field, Metric


class SingleLabelClassification(Metric):
    """
    Evaluate the accuracy, precision, recall, and F1 score of a single-label classification task.
    """

    def __init__(
        self,
        classes: Union[int, Set[str]],
        average: Literal["micro", "macro", "weighted"] = "macro",
    ):
        super().__init__(is_cpu_bound=True)
        assert average in ["macro", "micro", "weighted"]
        self._average = average
        if isinstance(classes, int):
            # Assume classes are 0, 1, 2, ..., classes-1
            self._classes = list(range(classes))
        else:
            self._classes = (
                classes if isinstance(classes, list) else set(classes)
            )
            if len(self._classes) != len(classes):
                raise ValueError("Classes must be unique")
        # Create a mapping from class labels to integers
        self._class_to_index = {
            label: index for index, label in enumerate(self._classes)
        }

    def compute(
        self,
        predicted_class: Union[str, int, List[float]],
        ground_truth_class: Union[str, int],
    ):
        if isinstance(predicted_class, list):
            predicted_class = np.argmax(
                predicted_class
            ).item()  # Convert to int
            if self._classes is not None:
                predicted_class = self._classes[predicted_class]
        return {
            "classification_prediction": predicted_class,
            "classification_ground_truth": ground_truth_class,
            "classification_correct": predicted_class == ground_truth_class,
        }

    def aggregate(
        self,
        results: List[Dict[str, Union[str, int]]],
    ) -> Any:
        pred = np.array(
            [
                self._class_to_index[r["classification_prediction"]]
                for r in results
            ]
        )
        gt = np.array(
            [
                self._class_to_index[r["classification_ground_truth"]]
                for r in results
            ]
        )

        accuracy = accuracy_score(gt, pred)
        balanced_accuracy = balanced_accuracy_score(gt, pred)
        precision = precision_score(
            gt, pred, average=self._average, zero_division=1.0
        )
        recall = recall_score(
            gt, pred, average=self._average, zero_division=1.0
        )
        f1 = f1_score(gt, pred, average=self._average, zero_division=1.0)

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @property
    def schema(self):
        return {
            "classification_prediction": Field(type=Union[str, int]),
            "classification_ground_truth": Field(type=Union[str, int]),
            "classification_correct": Field(type=bool),
        }
