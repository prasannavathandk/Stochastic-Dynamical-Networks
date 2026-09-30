import pytest

from ReductionFramework.Errors import ValueErrorCode
from ReductionFramework.GraphModelSpec import ConfigurationModelSpec


def Test_configuration_model_spec_is_imported():
	"""Ensure the configuration model specification is available to pytest."""
	assert ConfigurationModelSpec is not None
	
def Test_valid_instantiation():
    """Test that a valid ConfigurationModelSpec can be instantiated."""
    spec = ConfigurationModelSpec(
        num_nodes=3,
        num_edges=3,
        inDegreeSeq=[1, 1, 1],
        outDegreeSeq=[1, 1, 1],
        birthRateSeq=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        deathRateSeq=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    assert spec.num_nodes == 3
    assert spec.num_edges == 3
    assert spec.inDegreeSeq == [1, 1, 1]
    assert spec.outDegreeSeq == [1, 1, 1]
    assert spec.birthRateSeq == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert spec.deathRateSeq == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

@pytest.mark.parametrize(
        "num_nodes, num_edges, inDegreeSeq, outDegreeSeq, birthRateSeq, deathRateSeq", "error_code",
        [
            (3, 3, [1, 1], [1, 1, 1], [0.1] * 9, [0.1] * 9, "IN_DEG_LEN_MISMATCH"),
            (3, 3, [1, 1, 1], [1, 1], [0.1] * 9, [0.1] * 9, "OUT_DEG_LEN_MISMATCH"),
            
            (3, 3, [1, 0, 1], [1, 1, 1], [0.1] * 9, [0.1] * 9, "IN_DEG_POS_INT_MISMATCH"),
            (3, 3, [1, -1, 1], [1, 1, 1], [0.1] * 9, [0.1] * 9, "IN_DEG_POS_INT_MISMATCH"),

            (3, 3, [1, 1, 1], [1, 0, 1], [0.1] * 9, [0.1] * 9, "OUT_DEG_POS_INT_MISMATCH"),
            (3, 3, [1, 1, 1], [1, -1, 1], [0.1] * 9, [0.1] * 9, "OUT_DEG_POS_INT_MISMATCH"),

            (3, 3, [1, 1, 2], [1, 1, 1], [0.1] * 9, [0.1] * 9, "DEG_SUM_MISMATCH"),

            (3, 3, [1, 1, 1], [1, 1, 1], [0.1] * 8, [0.1] * 9, "BIRTH_RATE_LEN_MISMATCH"),
            (3, 3, [1, 1, 1], [1, 1, 1], [-0.1] * 9, [0.1] * 9, "BIRTH_RATE_NEGATIVE"),

            (3, 3, [1, 1, 1], [1, 1, 1], [0.1] * 9, [0.1] * 8, "DEATH_RATE_LEN_MISMATCH"),
            (3, 3, [1, 1, 1], [1, 1, 1], [0.1] * 9, [-0.1] * 9, "DEATH_RATE_NEGATIVE")
        ]
    )

def Test_parameters(
        num_nodes, num_edges, inDegreeSeq, outDegreeSeq, birthRateSeq, deathRateSeq, error_code
    ):
    """Test that invalid parameters raise the appropriate ValueErrorCode."""
    with pytest.raises(ValueErrorCode) as exc_info:
        ConfigurationModelSpec(
            num_nodes=num_nodes,
            num_edges=num_edges,
            inDegreeSeq=inDegreeSeq,
            outDegreeSeq=outDegreeSeq,
            birthRateSeq=birthRateSeq,
            deathRateSeq=deathRateSeq
        )
    assert exc_info.value.code == error_code


