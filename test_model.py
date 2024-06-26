import torch
from typing import Any
from pymlab.test import TestResults


async def test_model(
    dataset_path: str,
    parameters: dict[str, Any],
    result_id: str,
    trained_model: str,
    **kwargs
) -> TestResults:
    return TestResults(
        metrics={},
        predictions=[],
        files={}
    )