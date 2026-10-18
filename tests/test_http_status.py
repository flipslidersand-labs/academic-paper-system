"""Truth table for academic_paper.http_status (#739)."""

import pytest

from academic_paper.http_status import is_client_error, is_retryable_status


@pytest.mark.parametrize(
    "status,client,retryable",
    [
        (None, False, False),
        (200, False, False),
        (399, False, False),
        (400, True, False),
        (404, True, False),
        (429, True, True),
        (499, True, False),
        (500, False, True),
        (503, False, True),
    ],
)
def test_status_truth_table(status, client, retryable):
    assert is_client_error(status) is client
    assert is_retryable_status(status) is retryable
