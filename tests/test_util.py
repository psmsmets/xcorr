import pytest
import numpy as np
from numpy.testing import assert_allclose

from xcorr import util


def test_to_seconds:
    out = util.to_seconds(1)
    assert out == 1

    out = util.to_seconds(util._one_second)
    assert out == 1.
