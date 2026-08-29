import pandas as pd

from src.analysis import calibration_table, population_stability_index


def test_calibration_table_preserves_observations():
    table = calibration_table([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9])
    assert table["observations"].sum() == 4


def test_psi_is_zero_for_identical_distribution():
    values = pd.Series([1, 2, 3, 4, 5] * 10)
    assert population_stability_index(values, values) == 0.0
