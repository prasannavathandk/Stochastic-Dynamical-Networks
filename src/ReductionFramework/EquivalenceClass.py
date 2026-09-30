from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Generic, Hashable, Iterable, Protocol, Sequence, TypeVar

import numpy as np
import scipy as sp

from ReductionFramework.GraphModelSpec import GraphSpec, StateT, PermutationState, MatrixState

KeyT = TypeVar("KeyT", bound=Hashable)

class TransitionMatrix[StateT]():
    space: Sequence[StateT]
    transition_fn: Callable[[StateT, StateT, GraphSpec[StateT]], float]

    def size(self) -> tuple[int, int]:
        """Returns the size of the transition matrix."""
        return len(self.space), len(self.space)

    def get_transition_matrix(
        self,
        spec: GraphSpec[StateT],
    ) -> np.ndarray:
        """Returns the transition matrix as a 2D numpy array."""
        size = self.size()
        matrix = np.zeros(size, dtype=np.float64)
        for i, state_from in enumerate(self.space):
            for j, state_to in enumerate(self.space):
                matrix[i, j] = self.transition_fn(state_from, state_to, spec)
        return matrix

def totalCount_fn(state: PermutationState, spec: GraphSpec[PermutationState], matrixState = False) -> int:
    """A simple partitioning scheme that returns the total number of edges in the state."""
    if matrixState:
        return sum(spec.matrix_from_permutation(state).flatten())
    else:
        return sum(1 for x in state if x != -1)

def blockCount_fn(state: PermutationState, spec: GraphSpec[PermutationState], matrixState = False) -> tuple[int, ...]:
    """A partitioning scheme that sums the edges in blocks of a given size."""
    if matrixState:
        return tuple(
            sum(spec.matrix_from_permutation(state)[rows[0]:rows[1], cols[0]:cols[1]].flatten())
            for rows, cols in spec.blocks
        )
    else:
        return tuple(
            sum(
                1
                for i, j in enumerate(state)
                if j != -1
                and rows[0] <= i < rows[1]
                and cols[0] <= j < cols[1]
            )
            for rows, cols in spec.blocks
        )

class PartitionScheme(Protocol[StateT, KeyT]):
    def key(
        self,
        state: StateT,
        spec: GraphSpec[StateT],
    ) -> KeyT:
        ...

    def keys(
        self,
        spec: GraphSpec[StateT],
    ) -> Iterable[KeyT]:
        ...

    def size(
        self,
        key: KeyT,
        spec: GraphSpec[StateT],
    ) -> int:
        ...

    def get_partition(
        self,
        key: KeyT,
        spec: GraphSpec[StateT],
    ) -> Iterator[StateT]:
        ...

    def generate_partitions(
        self,
        spec: GraphSpec[StateT],
    ) -> dict[KeyT, list[StateT]]:
        ...

@dataclass(frozen=True, slots=True)
class CountPartitionScheme(Generic[StateT, KeyT]):
    key_fn: Callable[
        [StateT, GraphSpec[StateT]],
        KeyT,
    ]
    
    def key(
        self,
        state: StateT,
        spec: GraphSpec[StateT],
    ) -> KeyT:
        return self.key_fn(state, spec)

    def keys(
        self,
        spec: GraphSpec[StateT],
    ) -> Iterable[KeyT]:
        return {
            self.key(state, spec) 
            for state in spec.generate_states()
        }

    def size(
        self,
        key: KeyT,
        spec: GraphSpec[StateT],
    ) -> int:
        return sum(
            1
            for state in spec.generate_states()
            if self.key(state, spec) == key
        )

    def get_partition(
        self,
        key: KeyT,
        spec: GraphSpec[StateT],
    ) -> Iterator[StateT]:
        for state in spec.generate_states():
            if self.key(state, spec) == key:
                yield state

    def generate_partitions(
        self,
        spec: GraphSpec[StateT],
    ) -> dict[KeyT, Iterable[StateT]]:
        partitions: defaultdict[KeyT, set[StateT]] = defaultdict(set)

        for state in spec.generate_states():
            partitions[self.key(state, spec)].add(state)
        return dict(partitions)

@dataclass(frozen=True, slots=True)
class AnalyticalCountPartitionScheme(Generic[StateT]):
    key_fn: Callable[
        [StateT, GraphSpec[StateT]],
        int,
    ] = totalCount_fn
    
    def key(
        self,
        state: StateT,
        spec: GraphSpec[StateT],
    ) -> int:
        return self.key_fn(state, spec)

    def keys(
        self,
        spec: GraphSpec[StateT],
    ) -> Iterable[int]:
        return list(range(0, spec.num_edges + 1))

    def size(
        self,
        key: int,
        spec: GraphSpec[StateT],
    ) -> int:
        n = spec.num_edges
        z = key
        return int((sp.special.comb(n , n - z)/sp.special.factorial(n - z))*sp.special.factorial(n))

    def get_partition(
        self,
        key: int,
        spec: GraphSpec[StateT],
    ) -> Iterator[StateT]:
        raise NotImplementedError("AnalyticalCountPartitionScheme does not support generating partitions.")

    def generate_partitions(
        self,
        spec: GraphSpec[StateT],
    ) -> dict[int, Iterable[StateT]]:
        raise NotImplementedError("AnalyticalCountPartitionScheme does not support generating partitions.")

@dataclass(frozen=True, slots=True)
class InvariantSet(Generic[StateT, KeyT]):
    spec: GraphSpec[StateT]
    scheme: PartitionScheme[StateT, KeyT]

    def key(self, state: StateT) -> KeyT:
        return self.scheme.key(state, self.spec)

    def keys(self) -> Iterable[KeyT]:
        return self.scheme.keys(
            self.spec
        )

    def size(self, key: KeyT) -> int:
        return self.scheme.size(
            key,
            self.spec,
        )

    def partition(self, key: KeyT) -> Iterator[StateT]:
        return self.scheme.get_partition(
            key,
            self.spec,
        )

    def partitions(self) -> dict[KeyT, list[StateT]]:
        try:
            return self.scheme.generate_partitions(
                self.spec,
            )
        except Exception as e:
            raise e.with_traceback(None)
