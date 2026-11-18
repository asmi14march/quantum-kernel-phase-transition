"""Batched statevector engine.

A feature map is a Python function ``fmap(engine, x)`` that issues gate calls such as
``engine.h(q)`` or ``engine.p(theta, q)``. The *same* function is executed by

* :class:`QiskitBuilder`   -> a parameterised ``qiskit.QuantumCircuit`` (``x`` is a ParameterVector),
* :class:`BatchSimulator`  -> statevectors for a whole dataset at once (``x[i]`` is a NumPy column).

Evolving all N samples together turns N separate circuit simulations into a handful of
vectorised tensor operations; ``tests/test_simulator.py`` checks the result against
``qiskit.quantum_info.Statevector`` to machine precision, including Qiskit's little-endian
qubit ordering.
"""
from __future__ import annotations

import numpy as np


class BatchSimulator:
    """Simulate one circuit structure on B different parameter vectors simultaneously."""

    def __init__(self, n_qubits: int, batch: int):
        self.n = n_qubits
        self.B = batch
        self.psi = np.zeros((batch,) + (2,) * n_qubits, dtype=np.complex128)
        self.psi[(slice(None),) + (0,) * n_qubits] = 1.0

    # Qiskit is little-endian: qubit q is bit q of the basis index, i.e. C-order axis n-1-q.
    def _ax(self, q):
        return 1 + (self.n - 1 - q)

    def _sl(self, q, v):
        s = [slice(None)] * (self.n + 1)
        s[self._ax(q)] = v
        return tuple(s)

    def _bcast(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.ndim == 0:
            theta = np.full(self.B, float(theta))
        return theta.reshape((self.B,) + (1,) * (self.n - 1))

    # ---- gates -------------------------------------------------------------------
    def h(self, q):
        a0, a1 = self.psi[self._sl(q, 0)].copy(), self.psi[self._sl(q, 1)].copy()
        r = 1 / np.sqrt(2)
        self.psi[self._sl(q, 0)] = r * (a0 + a1)
        self.psi[self._sl(q, 1)] = r * (a0 - a1)

    def p(self, theta, q):
        self.psi[self._sl(q, 1)] *= np.exp(1j * self._bcast(theta))

    def rz(self, theta, q):
        t = self._bcast(theta)
        self.psi[self._sl(q, 0)] *= np.exp(-0.5j * t)
        self.psi[self._sl(q, 1)] *= np.exp(0.5j * t)

    def ry(self, theta, q):
        t = self._bcast(theta)
        c, s = np.cos(t / 2), np.sin(t / 2)
        a0, a1 = self.psi[self._sl(q, 0)].copy(), self.psi[self._sl(q, 1)].copy()
        self.psi[self._sl(q, 0)] = c * a0 - s * a1
        self.psi[self._sl(q, 1)] = s * a0 + c * a1

    def rx(self, theta, q):
        t = self._bcast(theta)
        c, s = np.cos(t / 2), np.sin(t / 2)
        a0, a1 = self.psi[self._sl(q, 0)].copy(), self.psi[self._sl(q, 1)].copy()
        self.psi[self._sl(q, 0)] = c * a0 - 1j * s * a1
        self.psi[self._sl(q, 1)] = -1j * s * a0 + c * a1

    def _sl2(self, q1, v1, q2, v2):
        s = [slice(None)] * (self.n + 1)
        s[self._ax(q1)] = v1
        s[self._ax(q2)] = v2
        return tuple(s)

    def cx(self, c, t):
        i, j = self._sl2(c, 1, t, 0), self._sl2(c, 1, t, 1)
        tmp = self.psi[i].copy()
        self.psi[i] = self.psi[j]
        self.psi[j] = tmp

    def cz(self, a, b):
        self.psi[self._sl2(a, 1, b, 1)] *= -1

    def statevectors(self) -> np.ndarray:
        return self.psi.reshape(self.B, -1)


class _Columns:
    """Lets feature-map code write x[i] for both NumPy batches and Qiskit ParameterVectors."""

    def __init__(self, X):
        self.X = X

    def __getitem__(self, i):
        return self.X[:, i]


class QiskitBuilder:
    """Records the same gate calls into a qiskit.QuantumCircuit."""

    def __init__(self, n_qubits: int):
        from qiskit import QuantumCircuit
        self.qc = QuantumCircuit(n_qubits)

    def h(self, q): self.qc.h(q)
    def p(self, t, q): self.qc.p(t, q)
    def rz(self, t, q): self.qc.rz(t, q)
    def ry(self, t, q): self.qc.ry(t, q)
    def rx(self, t, q): self.qc.rx(t, q)
    def cx(self, c, t): self.qc.cx(c, t)
    def cz(self, a, b): self.qc.cz(a, b)


def simulate(fmap, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Statevectors |phi(x)> for every row of X, shape (N, 2^n)."""
    X = np.atleast_2d(np.asarray(X, float))
    n = fmap.n_qubits
    out = np.empty((len(X), 2**n), dtype=np.complex128)
    # keep each batch tensor cache-sized (~2^17 amplitudes); large batches become memory-bound
    chunk = max(1, min(chunk, int(2**17 // 2**n)))
    for s in range(0, len(X), chunk):
        Xb = X[s:s + chunk]
        eng = BatchSimulator(n, len(Xb))
        fmap.build(eng, _Columns(Xb))
        out[s:s + chunk] = eng.statevectors()
    return out


def to_qiskit(fmap):
    """Parameterised Qiskit circuit for a feature map (for transpilation / hardware runs)."""
    from qiskit.circuit import ParameterVector
    x = ParameterVector("x", fmap.n_features)
    b = QiskitBuilder(fmap.n_qubits)
    fmap.build(b, x)
    return b.qc


def simulate_qiskit(fmap, X: np.ndarray) -> np.ndarray:
    """Reference path: one qiskit.quantum_info.Statevector per sample (slow, for validation)."""
    from qiskit.quantum_info import Statevector
    qc = to_qiskit(fmap)
    return np.array([Statevector(qc.assign_parameters(np.asarray(x, float))).data for x in X])
