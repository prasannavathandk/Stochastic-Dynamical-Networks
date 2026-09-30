from ReductionFramework.GraphModelSpec import ConfigurationModelSpec
from ReductionFramework.EquivalenceClass import AnalyticalCountPartitionScheme, CountPartitionScheme, InvariantSet, blockCount_fn, totalCount_fn
from ReductionFramework.TransitionMatrix import BoltzmannTransition_Scheme, FastTransitionMatrix, UniformTransition_Scheme
from ReductionFramework.models.Numerical import CoStochNet

if __name__ == "__main__":
    # Example usage of ComStochNet
    inDegreeSeq = [1, 2]
    outDegreeSeq = [2, 1]
    birthRateSeq = [0.1]*len(inDegreeSeq)**2
    deathRateSeq = [0.05]*len(inDegreeSeq)**2

    CMSpec = ConfigurationModelSpec(
        inDegreeSeq=inDegreeSeq,
        outDegreeSeq=outDegreeSeq
    )

    print(CMSpec.num_nodes)
    print(CMSpec.num_edges)
    print(CMSpec.inDegreeSeq)
    print(CMSpec.outDegreeSeq)
    print(CMSpec.degreePairs)
    print(CMSpec.maxEdges)
    print(CMSpec.ensembleSize)
    print(CMSpec.blocks)
    print()

    #print(len(set(CMSpec.generate_states())))
    
    eqClass = InvariantSet(
        spec=CMSpec,
        scheme=CountPartitionScheme(
            key_fn=totalCount_fn
        )
    )
    print(eqClass.keys())
    print([eqClass.size(k) for k in eqClass.keys()])
    print()

    transitionMatrix = FastTransitionMatrix(
        eqClass=eqClass,
        transition_scheme=BoltzmannTransition_Scheme(
            spec=CMSpec
        )
    )

    print(transitionMatrix.size(list(eqClass.keys())[-1]))
    print(transitionMatrix.get_transition_matrix(list(eqClass.keys())[-1]))
    print(transitionMatrix.generate_transition_matrices())
    print(transitionMatrix.get_stationary_distribution(list(eqClass.keys())[-1]))
    print(transitionMatrix.generate_stationary_distributions())
    print()
    #print(partitioning.partitions())

    #network = CoStochNet(inDegreeSeq, outDegreeSeq, birthRateSeq, deathRateSeq)
    #network.build(method='L1')