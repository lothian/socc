"""
Unit and regression test for the socc package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import socc


def test_socc_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "socc" in sys.modules
