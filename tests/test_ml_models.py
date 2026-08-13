"""Basic tests for ml_models.CrimeMLModels."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models import CrimeMLModels

REGRESSION_MODEL_NAMES = {
    "Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree",
    "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM",
}
CLASSIFICATION_MODEL_NAMES = {
    "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM",
}
CLUSTERING_MODEL_NAMES = {"KMeans", "Agglomerative"}


@pytest.fixture
def ml():
    return CrimeMLModels()


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    n = 60
    X = pd.DataFrame({
        "feature_1": rng.normal(size=n),
        "feature_2": rng.normal(size=n),
        "feature_3": rng.normal(size=n),
    })
    y = 3 * X["feature_1"] - 2 * X["feature_2"] + rng.normal(scale=0.1, size=n)
    return train_test_split(X, y, test_size=0.25, random_state=42)


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(7)
    n = 60
    X = pd.DataFrame({
        "feature_1": rng.normal(size=n),
        "feature_2": rng.normal(size=n),
    })
    y = (X["feature_1"] + X["feature_2"] > 0).astype(int)
    return train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)


def test_train_regression_models_returns_all_models(ml, regression_data):
    X_train, X_test, y_train, y_test = regression_data

    results, best_model_name = ml.train_regression_models(X_train, X_test, y_train, y_test)

    assert set(results.keys()) == REGRESSION_MODEL_NAMES
    assert best_model_name in REGRESSION_MODEL_NAMES
    for name, r in results.items():
        assert "test_r2" in r
        assert "predictions" in r
        assert len(r["predictions"]) == len(y_test)
    assert ml.results["regression"] is results


def test_train_classification_models_returns_all_models(ml, classification_data):
    X_train, X_test, y_train, y_test = classification_data

    results, best_model_name = ml.train_classification_models(X_train, X_test, y_train, y_test)

    assert set(results.keys()) == CLASSIFICATION_MODEL_NAMES
    assert best_model_name in CLASSIFICATION_MODEL_NAMES
    for name, r in results.items():
        assert 0.0 <= r["test_accuracy"] <= 1.0
        assert r["confusion_matrix"].shape == (2, 2)


def test_train_clustering_models_returns_expected_keys(ml):
    rng = np.random.default_rng(1)
    cluster_a = rng.normal(loc=0, scale=0.5, size=(20, 2))
    cluster_b = rng.normal(loc=10, scale=0.5, size=(20, 2))
    X = pd.DataFrame(np.vstack([cluster_a, cluster_b]), columns=["x", "y"])

    results, best_model_name = ml.train_clustering_models(X, n_clusters=2)

    assert set(results.keys()) == CLUSTERING_MODEL_NAMES
    assert best_model_name in CLUSTERING_MODEL_NAMES
    for name, r in results.items():
        assert len(r["labels"]) == len(X)
        assert -1.0 <= r["silhouette_score"] <= 1.0


def test_apply_pca_shapes(ml):
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(50, 6)))

    X_pca, pca, variance_ratio, cumulative_variance = ml.apply_pca(X, n_components=3)

    assert X_pca.shape == (50, 3)
    assert len(variance_ratio) == 3
    assert cumulative_variance[-1] <= 1.0 + 1e-9
    assert np.all(np.diff(cumulative_variance) >= -1e-9)  # non-decreasing


def test_save_and_load_model_roundtrip(ml, tmp_path, regression_data):
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = regression_data
    model = LinearRegression().fit(X_train, y_train)
    filename = tmp_path / "model.pkl"

    ml.save_model(model, str(filename))
    loaded = ml.load_model(str(filename))

    np.testing.assert_allclose(model.predict(X_test), loaded.predict(X_test))
