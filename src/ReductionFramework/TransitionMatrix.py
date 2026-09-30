from abc import ABC, abstractmethod
from collections import defaultdict
from collections import abc
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Annotated, Generic, Hashable, Iterable, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from ReductionFramework.GraphModelSpec import ConfigurationModelSpec, GraphSpec, StateT, PermutationState, MatrixState
from ReductionFramework.EquivalenceClass import InvariantSet, KeyT


@dataclass(frozen=True, slots=True)
class Transition_scheme(ABC, Generic[StateT]):

    def size(
        self,
        space: Sequence[StateT]
    ) -> int:
        """Returns the dimension of the transition matrix."""
        n = len(space)
        if n == 0:
            raise ValueError(
                "State space cannot be empty."
            )
        return n

    def validate(
            self,
            matrix: NDArray[np.float64],
            atol: float = 1e-8
    ) -> None:

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "Transition matrix must be square."
            )
        
        n = matrix.shape[0]

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "Transition matrix contains non-finite entries."
            )

        if np.any(matrix < -atol):
            raise ValueError(
                "Transition matrix contains negative entries."
            )
    
        """Validates that the transition matrix is a valid stochastic matrix."""

        
        if not np.allclose(matrix.sum(axis=1), 1, atol=atol):
            raise ValueError(
                "Transition matrix rows do not sum to 1."
            )

        """Validate that the markoc chain for the transition matrix is irreducible.

        # Treat P as the adjacency matrix of a directed graph..  
        # The Markov chain is irreducible iff this graph is strongly connected.
        """
        adjacency = matrix > atol

        def reachable(graph: NDArray[np.bool_]) -> set[int]:
            visited: set[int] = set()
            pending = [0]

            while pending:
                node = pending.pop()
                if node in visited:
                    continue
                visited.add(node)
                pending.extend(
                    np.flatnonzero(
                        graph[node]
                    ).tolist()
                )
            return visited

        if (
            len(reachable(adjacency)) != n
            or len(reachable(adjacency.T)) != n
        ):
            raise ValueError(
                "Transition matrix is not irreducible."
            )

        """Validate that the markoc chain for the transition matrix is aperiodic.

        # Since irreducibility has already been established,
        # a positive self-transition at any state guarantees aperiodicity.
        """
        if not np.any(
            np.diag(matrix) > atol
        ):
            raise ValueError(
                "Aperiodicity could not be established "
                "from a positive self-transition."
            )

    @abstractmethod
    def get_transition_matrix(
        self,
        space: Sequence[StateT]
    ) -> NDArray[np.float64]:
        """Returns the transition matrix as a 2D numpy array."""
        ...

    def get_stationary_distribution(
        self,   
        space: Sequence[StateT]
    ) -> NDArray[np.float64]:
        """Returns the stationary distribution of the transition matrix."""
        matrix = self.get_transition_matrix(space)

        self.validate(
            matrix
        )

        eigvals, eigvecs = np.linalg.eig(matrix.T)

        mask = np.isclose(
            eigvals,
            1.0,
        )

        if np.count_nonzero(mask) != 1:
            raise ValueError(
                "Expected a unique stationary distribution. The Equilavence Class may not be irreducible."
            )
        
        stationary = np.real(
            eigvecs[:, mask][:, 0]
        )

        # Eigenvectors are defined up to sign.
        if stationary.sum() < 0:
            stationary = -stationary

        stationary /= stationary.sum()

        return stationary.flatten()

@dataclass(frozen=True, slots=True)
class BoltzmannTransition_Scheme(
    Transition_scheme[PermutationState]
):
    spec: GraphSpec[PermutationState]    
    norm: Callable[[NDArray], float] = np.linalg.norm #Defualt is Frobenius Norm
    inverse_temperature: float = 1.0
    bias: float = 0.0
    weights: NDArray[np.float64] | None = None

    def energy_fn(self, state_from: PermutationState, state_to: PermutationState, weights: NDArray[np.float64]) -> float:
        distance =  self.norm(
                                self.spec.matrix_from_permutation(state_from) - self.spec.matrix_from_permutation(state_to)
                            )
        return distance + self.bias * self.norm(np.multiply(weights, self.spec.matrix_from_permutation(state_to)))
    
    def get_transition_matrix(
        self,
        space: Sequence[PermutationState],
        weights: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:

        n = self.size(space)

        matrix = np.zeros(
            (n, n),
            dtype=np.float64,
        )

        if self.weights is None:
            weights = np.ones(self.spec.num_edges)
        else:
            weights = self.weights

        for i, state_from in enumerate(space):
            for j, state_to in enumerate(space):

                energy = self.energy_fn(
                    state_from,
                    state_to,
                    weights
                )

                matrix[i, j] = np.exp(
                    -self.inverse_temperature * energy
                )

        row_sums = matrix.sum(
            axis=1,
            keepdims=True,
        )

        if np.any(row_sums == 0):
            raise ValueError(
                "Boltzmann kernel contains a zero-sum row."
            )

        matrix /= row_sums

        return matrix

@dataclass(frozen=True, slots=True)
class UniformTransition_Scheme(
    Transition_scheme[PermutationState]
):

    def get_transition_matrix(
        self,
        space: Sequence[StateT]
    ) -> NDArray[np.float64]:

        n = self.size(space)

        matrix = np.ones(
            (n, n),
            dtype=np.float64,
        )*(1/n)

        return matrix

    def get_stationary_distribution(
        self,
        space: Sequence[StateT]
    ) -> NDArray[np.float64]:

        n = len(space)

        stationary = np.ones(
            (n,),
            dtype=np.float64,
        )*(1/n)

        return stationary
    
@dataclass(frozen=True, slots=True)
class MetropolisTransitionScheme(
    Transition_scheme[PermutationState]
):
    energy_fn: Callable[
        [PermutationState],
        float,
    ]

    inverse_temperature: float = 1.0

    def get_transition_matrix(
        self,
        space: Sequence[PermutationState]
    ) -> NDArray[np.float64]:

        n = self.size(space)

        if n == 0:
            raise ValueError(
                "State space cannot be empty."
            )

        if n == 1:
            return np.ones(
                (1, 1),
                dtype=np.float64,
            )

        energies = np.array(
            [
                self.energy_fn(state)
                for state in space
            ],
            dtype=np.float64,
        )

        matrix = np.zeros(
            (n, n),
            dtype=np.float64,
        )

        # Uniform proposal to every other state.
        proposal = 1.0 / (n - 1)

        for i in range(n):
            for j in range(n):

                if i == j:
                    continue

                delta_energy = (
                    energies[j] - energies[i]
                )

                acceptance = min(
                    1.0,
                    np.exp(
                        -self.inverse_temperature
                        * delta_energy
                    ),
                )

                matrix[i, j] = (
                    proposal * acceptance
                )

            # Rejected proposals remain at state i.
            matrix[i, i] = (
                1.0 - matrix[i].sum()
            )

        assert np.allclose(
            matrix.sum(axis=1),
            1.0,
        )

        return matrix

    def get_stationary_distribution(
        self,
        space: Sequence[PermutationState]
    ) -> NDArray[np.float64]:

        energies = np.array(
            [
                self.energy_fn(state)
                for state in space
            ],
            dtype=np.float64,
        )

        # Numerical stabilization.
        energies -= energies.min()

        weights = np.exp(
            -self.inverse_temperature * energies
        )

        return weights / weights.sum()
    
@dataclass(frozen=True, slots=True)
class FastTransitionMatrix(Generic[StateT, KeyT]):
    eqClass: InvariantSet[StateT, KeyT]
    transition_scheme: Transition_scheme[StateT]

    def size(self, key: KeyT) -> tuple[int, int]:
        """Returns the size of the transition matrix for a given key."""
        partition = set(self.eqClass.partition(key))
        return (len(partition), len(partition))

    def get_transition_matrix(
        self,
        key: KeyT,
        sort_by: Callable[[Iterable[StateT]], Sequence[StateT]] = sorted
    ) -> NDArray[np.float64]:
        """Returns the transition matrix for a given key as a 2D numpy array."""
        partition = sort_by(set(self.eqClass.partition(key)))
        return self.transition_scheme.get_transition_matrix(space = partition)

    def generate_transition_matrices(
        self,
        sort_by_key: Callable[[Iterable[KeyT]], Sequence[KeyT]] = sorted,
        sort_by_state: Callable[[Iterable[StateT]], Sequence[StateT]] = sorted
    ) -> NDArray[np.float64]:
        """Returns a dictionary of transition matrices for each key."""
        matrix = np.zeros((self.eqClass.spec.ensembleSize, self.eqClass.spec.ensembleSize), dtype=np.float64)

        keys = sort_by_key(
            self.eqClass.keys()
        )

        idx = 0
        for key in keys:
            m = self.get_transition_matrix(
                key,
                sort_by=sort_by_state,
            )
            s = m.shape[0]
            matrix[idx:idx+s, idx:idx+s] = m
            idx += s

        assert np.allclose(matrix.sum(axis=1), 1), "Transition matrix rows do not sum to 1."

        return matrix

    def get_stationary_distribution(
        self,
        key: KeyT,
        sort_by: Callable[[Iterable[StateT]], Sequence[StateT]] = sorted
    ) -> NDArray[np.float64]:
        """Returns the stationary distribution of the transition matrix."""
        partition = sort_by(set(self.eqClass.partition(key)))
        return self.transition_scheme.get_stationary_distribution(space=partition)

    def generate_stationary_distributions(
        self,
        sort_by_key: Callable[[Iterable[KeyT]], Sequence[KeyT]] = sorted,
        sort_by_state: Callable[[Iterable[StateT]], Sequence[StateT]] = sorted
    ) -> dict[KeyT, NDArray[np.float64]]:
        """Returns a dictionary of stationary distributions for each key."""
        keys = sort_by_key(
            self.eqClass.keys()
        )

        return {
            key: self.get_stationary_distribution(
                key,
                sort_by=sort_by_state,
            )
            for key in keys
        }