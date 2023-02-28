import numpy as np
import pytest

numbers_to_test = [4, 5, 1241531, 234532]


@pytest.mark.parametrize("N", numbers_to_test)
def test_basic(N: int):
    d1 = np.ones(N)
    d2 = np.ones(N)
    np.testing.assert_allclose(d1, d2)
