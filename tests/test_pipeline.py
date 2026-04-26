"""
Tests unitaires pour le pipeline ML Titanic.
Lancer avec : pytest tests/ -v
"""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_pipeline import train_model, evaluate_model, save_model, load_model


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X_train = np.random.rand(100, 4)
    y_train = np.random.randint(0, 2, 100)
    X_test  = np.random.rand(20, 4)
    y_test  = np.random.randint(0, 2, 20)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def trained_model(sample_data):
    X_train, _, y_train, _ = sample_data
    return train_model(X_train, y_train, kernel="rbf", C=1.0)


class TestTrainModel:
    def test_retourne_modele(self, sample_data):
        X_train, _, y_train, _ = sample_data
        assert train_model(X_train, y_train) is not None

    def test_kernel_linear(self, sample_data):
        X_train, _, y_train, _ = sample_data
        assert train_model(X_train, y_train, kernel="linear") is not None

    def test_erreur_si_none(self):
        with pytest.raises(ValueError):
            train_model(None, None)

    def test_erreur_si_vide(self):
        with pytest.raises(ValueError):
            train_model(np.array([]), np.array([]))


class TestEvaluateModel:
    def test_score_entre_0_et_1(self, sample_data, trained_model):
        _, X_test, _, y_test = sample_data
        score = evaluate_model(trained_model, X_test, y_test)
        assert 0.0 <= score <= 1.0

    def test_erreur_si_modele_none(self, sample_data):
        _, X_test, _, y_test = sample_data
        with pytest.raises(ValueError):
            evaluate_model(None, X_test, y_test)

    def test_erreur_si_donnees_none(self, trained_model):
        with pytest.raises(ValueError):
            evaluate_model(trained_model, None, None)


class TestSaveLoadModel:
    def test_save_cree_fichier(self, trained_model, tmp_path):
        filepath = str(tmp_path / "test_model.pkl")
        save_model(trained_model, filepath)
        assert os.path.exists(filepath)

    def test_load_retourne_modele(self, trained_model, tmp_path):
        filepath = str(tmp_path / "test_model.pkl")
        save_model(trained_model, filepath)
        assert load_model(filepath) is not None

    def test_save_erreur_si_none(self):
        with pytest.raises(ValueError):
            save_model(None, "model.pkl")

    def test_save_erreur_si_nom_vide(self, trained_model):
        with pytest.raises(ValueError):
            save_model(trained_model, "")

    def test_load_erreur_fichier_absent(self):
        with pytest.raises(FileNotFoundError):
            load_model("fichier_inexistant.pkl")

    def test_load_erreur_nom_vide(self):
        with pytest.raises(ValueError):
            load_model("")