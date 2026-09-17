import pytest

from mlops_demo.promote import passes_quality_gate


@pytest.mark.parametrize(
    ("candidate", "champion", "minimum", "expected"),
    [
        (0.95, None, 0.90, True),
        (0.89, None, 0.90, False),
        (0.94, 0.95, 0.90, False),
        (0.95, 0.95, 0.90, True),
        (0.96, 0.95, 0.90, True),
    ],
)
def test_quality_gate(
    candidate: float, champion: float | None, minimum: float, expected: bool
) -> None:
    accepted, _ = passes_quality_gate(candidate, champion, minimum)
    assert accepted is expected
