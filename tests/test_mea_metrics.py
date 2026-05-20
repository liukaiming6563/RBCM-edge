from __future__ import annotations

import numpy as np

from rbcm_edge.mea.metrics import modulation_index


def test_modulation_index_positive_and_negative() -> None:
    assert modulation_index(3.0, 1.0) > 0
    assert modulation_index(1.0, 3.0) < 0
    assert np.isfinite(modulation_index(0.0, 0.0))
