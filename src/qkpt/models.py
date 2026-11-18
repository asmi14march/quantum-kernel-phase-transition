"""scikit-learn compatible quantum-kernel SVM and the classical baseline zoo.

Every model is returned together with a hyper-parameter grid, and all of them are tuned
with the same ``GridSearchCV`` protocol, so comparisons are apples-to-apples.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .feature_maps import make_feature_map
from .kernels import fidelity_kernel, project_psd, shot_noise
from .simulator import simulate


class QuantumKernelSVC(ClassifierMixin, BaseEstimator):
    """SVM on a fidelity quantum kernel.

    Parameters
    ----------
    feature_map : 'Z' | 'ZZ-linear' | 'ZZ-full' | 'ZZ-circular' | 'HEA'
    reps        : number of feature-map layers
    bandwidth   : scale applied to standardised features before encoding (x -> bandwidth * z)
    C           : SVM regularisation
    shots       : None for exact fidelities, else number of measurement shots per kernel entry
    """

    def __init__(self, feature_map="ZZ-linear", reps=2, bandwidth=1.0, C=1.0, shots=None,
                 random_state=None):
        self.feature_map = feature_map
        self.reps = reps
        self.bandwidth = bandwidth
        self.C = C
        self.shots = shots
        self.random_state = random_state

    def _encode(self, X):
        return simulate(self.fmap_, self.scaler_.transform(X) * self.bandwidth)

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.rng_ = np.random.default_rng(self.random_state)
        self.scaler_ = StandardScaler().fit(X)
        self.fmap_ = make_feature_map(self.feature_map, X.shape[1], self.reps)
        self.psi_ = self._encode(X)
        K = fidelity_kernel(self.psi_)
        if self.shots is not None:
            K = project_psd(shot_noise(K, self.shots, self.rng_, symmetric=True))
        self.K_train_ = K
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(K, y)
        self.classes_ = self.svc_.classes_
        return self

    def kernel(self, X):
        K = fidelity_kernel(self._encode(np.asarray(X, float)), self.psi_)
        if self.shots is not None:
            K = shot_noise(K, self.shots, self.rng_, symmetric=False)
        return K

    def decision_function(self, X):
        return self.svc_.decision_function(self.kernel(X))

    def predict(self, X):
        return self.svc_.predict(self.kernel(X))


class PrecomputedStateKernelSVC(ClassifierMixin, BaseEstimator):
    """SVM whose kernel is the overlap of *physical* quantum states.

    X holds integer indices into a bank of states (e.g. TFIM ground states), so the model
    acts on genuine quantum data rather than classically encoded features.
    """

    def __init__(self, states=None, C=1.0):
        self.states = states
        self.C = C

    def _K(self, ia, ib):
        A, B = self.states[ia], self.states[ib]
        return np.abs(A.conj() @ B.T) ** 2

    def fit(self, X, y):
        self.idx_ = np.asarray(X).ravel().astype(int)
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(self._K(self.idx_, self.idx_), y)
        self.classes_ = self.svc_.classes_
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(self._K(np.asarray(X).ravel().astype(int), self.idx_))

    def predict(self, X):
        return self.svc_.predict(self._K(np.asarray(X).ravel().astype(int), self.idx_))


QK_GRID = {"bandwidth": [0.05, 0.1, 0.2, 0.4, 0.7, 1.0], "C": [0.1, 1, 10, 100]}


def make_model(name: str, seed: int = 0):
    """Return (estimator, param_grid) for a named model."""
    if name.startswith("QK-"):
        return QuantumKernelSVC(feature_map=name[3:], random_state=seed), QK_GRID
    if name == "Linear SVM":
        return (make_pipeline(StandardScaler(), SVC(kernel="linear")),
                {"svc__C": [0.01, 0.1, 1, 10, 100]})
    if name == "RBF SVM":
        return (make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]})
    if name in ("MLP", "MLP (raw spins)"):
        return (make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64),
                                                              max_iter=2000, random_state=seed)),
                {"mlpclassifier__alpha": [1e-4, 1e-2, 1.0]})
    raise ValueError(name)


QUANTUM_MODELS = ["QK-Z", "QK-ZZ-linear", "QK-ZZ-full", "QK-HEA"]
CLASSICAL_MODELS = ["Linear SVM", "RBF SVM", "MLP"]
