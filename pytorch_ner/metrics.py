import numpy as np
from typing import List, Dict, DefaultDict
from sklearn.metrics import f1_score


def calculate_metrics(
        metrics: DefaultDict[str, List[float]],
        loss: float,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        idx2label: Dict[int, str],
) -> DefaultDict[str, List[float]]:
    """
    Calculate metrics on epoch.
    """

    metrics['loss'].append(loss)

    f1_per_class = f1_score(y_true=y_true, y_pred=y_pred, labels=range(len(idx2label)), average=None)
    for cls, f1 in enumerate(f1_per_class):
        metrics[f'f1 {idx2label[cls]}'].append(f1)

    return metrics
