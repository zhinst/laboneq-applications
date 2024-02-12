import pytest
import numpy as np

from laboneq_library.analysis.amplitude_rabi import extract_rabi_amplitude


def test_extract_rabi_amplitude():
    # Data from Amplitude Rabi experiment. Input data rotated without PCA.
    data = np.array(
        [
            -0.00277166,
            0.00613112,
            0.03883595,
            0.09857108,
            0.15548143,
            0.24910771,
            0.33763663,
            0.44266485,
            0.52049004,
            0.6251271,
            0.73224575,
            0.82647127,
            0.88175776,
            0.95589651,
            0.98624188,
            0.99398839,
            0.99838702,
            0.9704031,
            0.93536926,
            0.86714271,
            0.80404261,
        ]
    )
    amplitudes = np.array(
        [
            0.0,
            0.02133995,
            0.04267991,
            0.06401986,
            0.08535982,
            0.10669977,
            0.12803973,
            0.14937968,
            0.17071964,
            0.19205959,
            0.21339955,
            0.2347395,
            0.25607945,
            0.27741941,
            0.29875936,
            0.32009932,
            0.34143927,
            0.36277923,
            0.38411918,
            0.40545914,
            0.42679909,
        ]
    )
    results = extract_rabi_amplitude(data=data, amplitudes=amplitudes)
    assert results.pi_amplitude.nominal_value == pytest.approx(0.3284004378016136)
    assert results.pi_amplitude.std_dev == pytest.approx(0.003508484402063912)

    assert results.pi2_amplitude.nominal_value == pytest.approx(0.16302037270062014)
    assert results.pi2_amplitude.std_dev == pytest.approx(0.0026318551966845296)

    # Test default fitting model
    assert results.model.model.func(results.pi_amplitude.nominal_value, **results.model.best_values) == pytest.approx(1.0020635710040655)
    assert results.model.model.func(results.pi2_amplitude.nominal_value, **results.model.best_values) == pytest.approx(0.49907399921173384)
