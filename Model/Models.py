from iForests import gif, eif
from iForests.FariCutForest import FairCutForest
from Scores.MBAS import MBAS
from Scores.SBAS import SBAS
from iForests.SciForest import SCiForest

class IFModel:
    def __init__(self, model_type, score_method,
                 n_tree, sample_size, contamination, level, k_planes, majority, threshold):
        self.model_type = model_type
        self.score_method = score_method
        self.n_tree = n_tree
        self.phi = sample_size
        self.contamination = contamination
        self.level = level
        self.k_planes = k_planes
        self.majority = majority
        self.threshold = threshold

    def _predict_score(self, clf, X, fit_needed=False):
        if fit_needed:
            clf.fit(X)
        if self.score_method == "MBAS":
            scores = clf.predict(X)
            return MBAS.predict(scores)
        elif self.score_method == "SBAS":
            path_len = clf.compute_paths_all_tree(X)
            return SBAS.predict(path_len, threshold=self.threshold, majority=self.majority)

    def evaluate_IF(self, X):
        clf = eif.iForest(X=X, ntrees=self.n_tree, sample_size=self.phi, ExtensionLevel=0)
        return self._predict_score(clf, X)

    def evaluate_EIF(self, X):
        clf = eif.iForest(X=X, ntrees=self.n_tree, sample_size=self.phi, ExtensionLevel=1)
        return self._predict_score(clf, X)

    def evaluate_GIF(self, X):
        clf = gif.iForest(X=X, ntrees=self.n_tree, sample_size=self.phi)
        return self._predict_score(clf, X)

    def evaluate_SciForest(self, X):
        clf = SCiForest(n_trees=self.n_tree, sample_size=self.phi, k_planes=self.k_planes, extension_level=self.level)
        return self._predict_score(clf, X, fit_needed=True)

    def evaluate_FairCutForest(self, X):
        clf = FairCutForest(n_trees=self.n_tree, sample_size=self.phi, k_planes=self.k_planes, extension_level=self.level)
        return self._predict_score(clf, X, fit_needed=True)

    def evaluate(self, X, y_true):
        model_map = {
            "Isolation Forest": self.evaluate_IF,
            "Extended Isolation Forest": self.evaluate_EIF,
            "Generalized Isolation Forest": self.evaluate_GIF,
            "SciForest": self.evaluate_SciForest,
            "FairCutForest": self.evaluate_FairCutForest
        }
        if self.model_type not in model_map:
            raise ValueError(f"Unknown model type: {self.model_type}")

        y_pred = model_map[self.model_type](X)
        metrics = PerformanceMetrics.compute_metrics(y_true, y_pred)
        return metrics



from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from Scores import PerformanceMetrics
from DeepModels import DeepModels


class BenchmarkModels:
    """
    Klasik ve derin öğrenme tabanlı benchmark anomali modelleri.
    """

    @staticmethod
    def run(X, y_true, selected_model, hyperparams):
        y_pred = None
        model_instance = None
        params = hyperparams

        # --- Sklearn modelleri --- #
        if selected_model == "Sklearn IF":
            model_instance = IsolationForest(
                n_estimators=params.get("n_trees", 100),
                contamination=params.get("contamination", 0.05),
                max_samples=params.get("max_samples", "auto"),
                random_state=42
            )
            model_instance.fit(X)
            y_pred = (model_instance.predict(X) == -1).astype(int)

        elif selected_model == "One-Class SVM":
            model_instance = OneClassSVM(
                kernel=params.get("kernel", "rbf"),
                nu=params.get("nu", 0.05),
                gamma=params.get("gamma", "scale")
            )
            model_instance.fit(X)
            y_pred = (model_instance.predict(X) == -1).astype(int)

        elif selected_model == "Local Outlier Factor":
            model_instance = LocalOutlierFactor(
                n_neighbors=params.get("n_neighbors", 20),
                contamination=params.get("contamination", 0.05),
                novelty=True  # SHAP uyumlu
            )
            model_instance.fit(X)
            y_pred = (model_instance.predict(X) == -1).astype(int)

        elif selected_model == "Elliptic Envelope":
            model_instance = EllipticEnvelope(
                contamination=params.get("contamination", 0.05),
                random_state=42
            )
            model_instance.fit(X)
            y_pred = (model_instance.predict(X) == -1).astype(int)

        # --- Derin modeller (DeepModels sınıfını kullan) --- #
        elif selected_model == "Autoencoder":
            y_pred, model_instance = DeepModels.run_autoencoder(
                X,
                y_true=y_true,
                epochs=params.get("epochs", 50),
                batch_size=params.get("batch_size", 32),
                learning_rate=params.get("learning_rate", 0.001)
            )

        elif selected_model == "VAE":
            y_pred, model_instance = DeepModels.run_vae(
                X,
                y_true=y_true,
                epochs=params.get("epochs", 50),
                batch_size=params.get("batch_size", 32),
                learning_rate=params.get("learning_rate", 0.001),
                latent_dim=params.get("latent_dim", 4)
            )

        elif selected_model == "DeepSVDD":
            y_pred, model_instance = DeepModels.run_deepsvdd(X, y_true, params=params)

        else:
            raise ValueError(f"Unsupported model: {selected_model}")

        # --- Performans metrikleri --- #
        metrics = PerformanceMetrics.compute_metrics(y_true, y_pred)
        return metrics, model_instance
