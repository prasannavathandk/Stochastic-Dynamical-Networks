from dataclasses import dataclass, field
from collections.abc import Iterable, Iterator, Sequence
from abc import ABC, abstractmethod
import itertools
from typing import Generic, List, TypeVar

import numpy as np
from numpy.typing import NDArray
import scipy as sp

from ReductionFramework.Errors import ValueErrorCode

StateT = TypeVar("StateT")
PermutationState = tuple[int, ...]
MatrixState = NDArray[np.int8]
Block = tuple[
    tuple[int, ...],  # row indices
    tuple[int, ...],  # column indices
]

class GraphSpec(ABC, Generic[StateT]):
    """Abstract specification of a graph model."""

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        pass

    @property
    @abstractmethod
    def num_edges(self) -> int:
        """Returns the number of edges in the graph."""
        pass

    @property
    @abstractmethod
    def ensembleSize(self) -> int:
        """Returns the number of edges in the graph."""
        pass

    @property
    @abstractmethod
    def degreePairs(self) -> np.ndarray:
        """Returns the degree pairs as a 2D numpy array of shape (num_nodes, 2)."""
        pass

    @property
    @abstractmethod
    def maxEdges(self) -> int:
        """Returns the maximum number of edges possible for each pair of nodes."""
        pass

    @property
    @abstractmethod
    def blocks(self) -> Sequence[Block]:
        """Returns the block indices of the graph model."""
        pass

    @abstractmethod
    def generate_states(self) -> Iterator[StateT]:
        """Generates all possible states of the graph model."""
        pass

@dataclass(frozen=True, slots=True)
class ConfigurationModelSpec(GraphSpec[PermutationState]):
    inDegreeSeq: Iterable[int]
    outDegreeSeq: Iterable[int]

    directed: bool = True
    allow_self_loops: bool = True
    allow_multiple_edges: bool = True

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:        

        # Check the in-degree are strictly positive integers
        if (self.inDegreeSeq is not None and any(d <= 0 for d in self.inDegreeSeq)):
            raise ValueErrorCode("IN_DEG_POS_INT_MISMATCH", "In-degree sequence must contain only positive integers")

        # Check the out-degree are strictly positive integers
        if (self.outDegreeSeq is not None and any(d <= 0 for d in self.outDegreeSeq)):
            raise ValueErrorCode("OUT_DEG_POS_INT_MISMATCH", "Out-degree sequence must contain only positive integers")

        # Check that the length of in-degree sequence equals the length of out-degree sequence
        if len(self.inDegreeSeq) != len(self.outDegreeSeq):
            raise ValueErrorCode("DEG_SUM_MISMATCH", "Length of in-degree sequence must equal the length of out-degree sequence and equals the number of nodes")
        
        # Check that the sum of in-degree sequence equals the sum of out-degree sequence and matches the number of edges
        if sum(self.inDegreeSeq) != sum(self.outDegreeSeq):
            raise ValueErrorCode("DEG_SUM_MISMATCH", "Sum of in-degree sequence must equal sum of out-degree sequence and equals the number of edges")

        N = len(self.inDegreeSeq)

        # # Check each node pair has an associated birth rate
        # if self.birthRateSeq is None or len(self.birthRateSeq) != N**2:
        #     raise ValueErrorCode("BIRTH_RATE_LEN_MISMATCH", "Length of Birth rate sequence must match number of pairs of nodes (nodes squared)")
        # # Check the birth rates are non-negative
        # if (self.birthRateSeq is not None and any(r < 0 for r in self.birthRateSeq)):
        #     raise ValueErrorCode("BIRTH_RATE_NEGATIVE", "Birth rate sequence must contain only non-negative values")

        # # Check each node pair has an associated death rate
        # if self.deathRateSeq is None or len(self.deathRateSeq) != N**2:
        #     raise ValueErrorCode("DEATH_RATE_LEN_MISMATCH", "Length of Death rate sequence must match number of pairs of nodes (nodes squared)")
        # # Check the death rates are non-negative
        # if (self.deathRateSeq is not None and any(r < 0 for r in self.deathRateSeq)):
        #     raise ValueErrorCode("DEATH_RATE_NEGATIVE", "Death rate sequence must contain only non-negative values")

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        return len(self.inDegreeSeq)

    @property
    def num_edges(self) -> int:
        """Returns the number of edges in the graph."""
        return sum(self.inDegreeSeq)

    @property
    def ensembleSize(self) -> int:
        """Returns the size of the ensemble of possible states."""
        return np.sum(
            [
                (
                    (sp.special.comb(self.num_edges , self.num_edges - z)/sp.special.factorial(self.num_edges - z))*
                    sp.special.factorial(self.num_edges)
                ) for z in range(self.num_edges+1)
            ]
        ).astype(int)

    @property
    def degreePairs(self) -> np.ndarray:
        """
            Returns the degree pairs as a 2D numpy array of shape (num_nodes, 2), where each row corresponds to a node and contains its in-degree and out-degree.
        """
        return np.array([(a, b) for a in self.outDegreeSeq for b in self.inDegreeSeq])

    @property
    def maxEdges(self) -> int:
        """
            Returns the maximum number of edges possible for each pair of nodes based on the configuration model specifications.
        """
        return np.array([min(pair) for pair in self.degreePairs]).reshape(self.num_nodes, self.num_nodes)

    @property
    def blocks(self) -> Sequence[Block]:
        b = list(itertools.product(self.outDegreeSeq, self.inDegreeSeq))
        blocksRange = list()
        curRow = 0
        curCol = 0
        N = self.num_nodes
        for i in range(len(b)):    
            block: Block = tuple(
                    [
                        tuple([curRow, curRow + int(b[i][0])]),
                        tuple([curCol, curCol + int(b[i][1])])
                    ]
                )
            blocksRange.append(block)
            curCol += int(b[i][1])
            if (i + 1) % N == 0 and i != 0:        
                curCol = 0
                curRow += int(b[i][0])

        return blocksRange

    def generate_states(self) -> Iterator[PermutationState]:
        """
            Generates all possible states of the graph model as permutations.
        """
        N = self.num_nodes
        n = self.num_edges

        # Generate all permutations of the nodes
        for permutation in itertools.permutations(range(n)):
            for k in range(n+1):
                for trim in itertools.combinations(range(n), k):
                    state = list(permutation)
                    for i in trim:
                        state[i] = -1

                    yield tuple(state)

    def matrix_from_permutation(self, permutation: Sequence[int]) -> MatrixState:
        """
            Convert a permutation p into its (quasi) permutation matrix P.
        """

        n = len(permutation)

        if any(i < -1 or i >= n for i in permutation):
            raise ValueError(
                f"Expected a permutation sequence of length {n} with values from 0,...,{n - 1}, "
                f"got {tuple(permutation)}."
            )

        matrix = np.zeros((n, n), dtype=np.int8)
        for i, j in enumerate(permutation):
            if j != -1:  # Assuming -1 indicates a missing value in the permutation
                matrix[i, j] = 1

        return matrix


