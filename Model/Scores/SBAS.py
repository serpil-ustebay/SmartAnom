from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef, balanced_accuracy_score
)
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

class SBAS:

    @staticmethod
    def custom_sigmoid(x, k=2):
        """
        Custom sigmoid transformation function.
        It maps input values into the range [0, 1], with x=0 centered at 0.5.
        The 'k' parameter controls the steepness of the curve.
        """
        # The sigmoid function smoothly maps real numbers to (0, 1)
        return 1 / (1 + np.exp(-k * x))



    @staticmethod
    def predict(path_lengths, threshold, majority):
        """
        Predict anomalies based on sigmoid-transformed path lengths.

        Parameters
        ----------
        path_lengths : ndarray of shape (n_samples, n_trees)
            The path lengths or anomaly scores obtained from the isolation-based trees.

        threshold : float
            Sigmoid threshold. Samples with sigmoid score below this value
            are considered potential anomalies.

        majority : int
            The minimum number of trees that must classify a sample as an anomaly
            for it to be labeled as anomalous.

        Returns
        -------
        pred : ndarray of shape (n_samples,)
            Binary prediction array, where 1 indicates anomaly and 0 indicates normal.
        """

        # Apply sigmoid transformation to the path lengths
        scores = SBAS.custom_sigmoid(path_lengths)

        # Mark samples below the threshold as anomalies (1)
        binary = (scores <= threshold).astype(int)

        # Count how many trees voted each sample as anomalous
        total_votes = np.sum(binary, axis=1)

        # If the number of anomaly votes exceeds 'majority', label as anomaly (1)
        pred = (total_votes > majority).astype(int)

        return pred
