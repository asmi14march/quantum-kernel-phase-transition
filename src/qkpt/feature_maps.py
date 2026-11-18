"""Quantum feature maps |phi(x)> = U(x)|0^n>, one qubit per input feature.

Each map is written once as a sequence of gate calls and therefore runs both on the
batched simulator and as a real Qiskit circuit (see ``simulator.py``).

=================  ===================================================================
``Z``              product-state map (no entanglement); Qiskit ``z_feature_map``.
                   Its kernel factorises into a product of cosines -> classically trivial.
``ZZ-linear``      Havlicek et al. (Nature 2019) second-order Pauli-Z evolution, nearest-
                   neighbour entanglement; identical to Qiskit ``zz_feature_map(..,'linear')``.
``ZZ-full``        same with all-to-all ZZ terms (``entanglement='full'``).
``HEA``            hardware-efficient data re-uploading: RY(x)RZ(x) layers + CZ chain;
                   native to IBM heavy-hex devices with no SWAP overhead on a line.
=================  ===================================================================

Inputs are expected to be standardised and multiplied by a *bandwidth* (scale) factor
before encoding; see ``kernels.py`` for why the bandwidth matters.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def _pairs(n, entanglement):
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if entanglement == "circular":
        return ([(n - 1, 0)] if n > 2 else []) + [(i, i + 1) for i in range(n - 1)]
    if entanglement == "full":
        return list(combinations(range(n), 2))
    raise ValueError(entanglement)


@dataclass(frozen=True)
class FeatureMap:
    kind: str
    n_features: int
    reps: int = 2
    entanglement: str = "linear"

    @property
    def n_qubits(self) -> int:
        return self.n_features

    @property
    def name(self) -> str:
        if self.kind == "ZZ":
            return f"ZZ-{self.entanglement}"
        return self.kind

    def build(self, eng, x):
        n = self.n_features
        if self.kind == "Z":
            for _ in range(self.reps):
                for q in range(n):
                    eng.h(q)
                for q in range(n):
                    eng.p(2.0 * x[q], q)
        elif self.kind == "ZZ":
            pairs = _pairs(n, self.entanglement)
            for _ in range(self.reps):
                for q in range(n):
                    eng.h(q)
                for q in range(n):
                    eng.p(2.0 * x[q], q)
                for i, j in pairs:
                    eng.cx(i, j)
                    eng.p(2.0 * (np.pi - x[i]) * (np.pi - x[j]), j)
                    eng.cx(i, j)
        elif self.kind == "HEA":
            for _ in range(self.reps):
                for q in range(n):
                    eng.ry(x[q], q)
                    eng.rz(x[q], q)
                for i, j in _pairs(n, "linear"):
                    eng.cz(i, j)
        else:
            raise ValueError(f"unknown feature map {self.kind}")


def make_feature_map(name: str, n_features: int, reps: int = 2) -> FeatureMap:
    """'Z', 'ZZ-linear', 'ZZ-full', 'ZZ-circular' or 'HEA'."""
    if name.startswith("ZZ"):
        return FeatureMap("ZZ", n_features, reps, name.split("-", 1)[1])
    return FeatureMap(name, n_features, reps)


FEATURE_MAPS = ["Z", "ZZ-linear", "ZZ-full", "HEA"]
