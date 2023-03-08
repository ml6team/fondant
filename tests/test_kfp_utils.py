"""
Test scripts for kfp helpers
"""
import pytest

from express.kfp_utils import parse_kfp_list


@pytest.mark.parametrize(
    "kfp_parsed_string, expected_output",
    [
        ("['item1', 'item2', 'item3']", ['item1', 'item2', 'item3']),
        ("[]", []),
        ("['']", ['']),
        ("['1', '2', '3']", ['1', '2', '3']),
    ],
)
def test_parse_kfp_list(kfp_parsed_string, expected_output):
    assert parse_kfp_list(kfp_parsed_string) == expected_output
