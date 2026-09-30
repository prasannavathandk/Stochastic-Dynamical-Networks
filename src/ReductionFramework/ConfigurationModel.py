from abc import abstractmethod
import numpy as np

from ReductionFramework import GraphModelSpec

class ReStochNet:
    """
        ReducedStochasticNetworks: Reduction methods for stochastic structural dynamical networks. 
        This class provides methods to analyze networks with stochastically changing connections using a Markov Jump Process formulation
        and reduction techniques from multi-scale methods.
    """
    def __init__(self,
                 modelSpec: GraphModelSpec):
        self.modelSpec = modelSpec

    def initializeNetworkTopology(self):
        """
            Initialize the network topology based on the configuration model.
            This method generates the adjacency matrix and identifies the blocks of the network based on the degree sequences.
        """
        pass

    @abstractmethod
    def stateSpace(self):
        pass

    @abstractmethod
    def partitions(self):
        pass

    @abstractmethod
    def transitionMatrix(self):
        pass

    @abstractmethod
    def infinitesimalGenerator(self):
        pass

    @abstractmethod
    def effectiveGenerator(self):
        pass

    def simulateCTMC(self, initialState: int, timeHorizon: float, numSimulations: int):
        """
            Simulate the Continuous-Time Markov Chain (CTMC) for the stochastic network.
            This method simulates the CTMC based on the infinitesimal generator and returns the state trajectories over time.
        """
        Q = self.infinitesimalGenerator()
        states = [initialState] * numSimulations
        times = [0.0] * numSimulations
        trajectories = [[] for _ in range(numSimulations)]

        for sim in range(numSimulations):
            currentState = initialState
            currentTime = 0.0
            while currentTime < timeHorizon:
                rate = -Q[currentState, currentState]
                if rate <= 0:
                    break
                waitTime = np.random.exponential(1 / rate)
                currentTime += waitTime
                if currentTime >= timeHorizon:
                    break
                nextStateProbabilities = Q[currentState, :] / rate
                nextStateProbabilities[currentState] = 0  # Exclude self-transition
                nextStateProbabilities /= np.sum(nextStateProbabilities)  # Normalize
                nextState = np.random.choice(len(Q), p=nextStateProbabilities)
                trajectories[sim].append((currentTime, nextState))
                currentState = nextState

        return trajectories




    