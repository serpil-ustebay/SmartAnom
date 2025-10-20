import numpy as np

class MBAS(object):
    @staticmethod
    def predict(path_lenghts, threshold=0.6):
        """
        Predict anomalies based on individual anomaly scores.

        Parameters
        ----------
        path_lenghts : array-like of shape (n_samples,)
            The anomaly scores or path lengths computed for each sample.

        threshold : float, default=0.6
            The cutoff value used to classify samples.
            If the score > threshold → sample is labeled as anomaly (1),
            otherwise it is considered normal (0).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Binary prediction array where 1 indicates anomaly and 0 indicates normal.
        """

        # Convert input scores to a NumPy array to ensure proper vectorized operations
        scores = np.asarray(path_lenghts)

        # Apply the threshold rule:
        # Samples with score greater than the threshold are labeled as anomalies (1)
        y_pred = (scores > threshold).astype(int)

        # Return binary anomaly predictions
        return y_pred
