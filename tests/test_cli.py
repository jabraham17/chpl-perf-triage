import pytest
from triage.cli import main


def test_main_returns_zero():
    # TODO: dummy test
    result = main([])
    assert result == 0
