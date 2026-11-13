from pathlib import Path
import sys
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "artifacts" / "customer_churn_model.joblib"


class WeightedGradientBoosting(ClassifierMixin, BaseEstimator):
    """Compatibility definition for the custom class stored in the joblib artifact."""

    def __init__(
        self, n_estimators=100, learning_rate=0.1, max_depth=3,
        min_samples_split=2, min_samples_leaf=1, subsample=1.0,
        max_features=None, random_state=42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        weights = compute_sample_weight(class_weight="balanced", y=y)
        self.estimator_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            max_features=self.max_features,
            random_state=self.random_state,
        )
        self.estimator_.fit(X, y, sample_weight=weights)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


# The notebook serialized this estimator as __main__.WeightedGradientBoosting.
# Register the compatible class before joblib unpickles the artifact.
sys.modules["__main__"].WeightedGradientBoosting = WeightedGradientBoosting


@__import__("functools").lru_cache(maxsize=1)
def load_artifact():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {ARTIFACT_PATH}. "
            "Place customer_churn_model.joblib in the artifacts folder."
        )
    artifact = joblib.load(ARTIFACT_PATH)
    required = {"model", "threshold", "features"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing)}")
    return artifact
