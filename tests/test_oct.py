"""Test for running OCT"""
import os
import sys

import pytest

import numpy as np
import QDYN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import model
import oct


@pytest.fixture
def oct_model(request):
    """Model for OCT of non-Hermitian Hamiltonian"""
    datadir = os.path.splitext(request.module.__file__)[0]
    w1     = 6000.0 # MHz
    w2     = 5900.0 # MHz
    wc     = 6200.0 # MHz
    wd     = 5932.5 # MHz
    alpha1 = -290.0 # MHz
    alpha2 = -310.0 # MHz
    g      =   70.0 # MHz
    n_qubit = 5
    n_cavity = 6
    kappa = list(np.arange(n_cavity) * 0.05)[1:-1] + [10000.0, ]  # MHz
    gamma = [0.012, 0.024, 0.033, 10000.0]  # MHz
    pulse = QDYN.pulse.Pulse.read(os.path.join(datadir, "pulse.guess"))
    pulse.config_attribs['is_complex'] = True
    gate = QDYN.gate2q.Gate2Q.read(os.path.join(datadir, 'target_gate.dat'),
                                   name='O', format='array')
    return model.transmon_model(
        n_qubit, n_cavity, w1, w2, wc, wd, alpha1, alpha2, g, gamma, kappa,
        lambda_a=0.1, pulse=pulse, dissipation_model='non-Hermitian',
        gate=gate, iter_stop=5)


@pytest.fixture
def oct_filter_model(request):
    """Model for optimization with a spectral filter."""
    datadir = os.path.splitext(request.module.__file__)[0]
    w1     = 6000.0 # MHz
    w2     = 5900.0 # MHz
    wc     = 6200.0 # MHz
    wd     = 5932.5 # MHz
    alpha1 = -290.0 # MHz
    alpha2 = -310.0 # MHz
    g      =   70.0 # MHz
    n_qubit = 5
    n_cavity = 6
    kappa = list(np.arange(n_cavity) * 0.05)[1:-1] + [10000.0, ]  # MHz
    gamma = [0.012, 0.024, 0.033, 10000.0]  # MHz
    pulse = QDYN.pulse.Pulse.read(os.path.join(datadir, "pulse.guess"),
                                  freq_unit='MHz')
    pulse.config_attribs['is_complex'] = True
    pulse.config_attribs['oct_spectral_filter'] = 'filter.dat'
    gate = QDYN.gate2q.Gate2Q.read(os.path.join(datadir, 'target_gate.dat'),
                                   name='O', format='array')
    return model.transmon_model(
        n_qubit, n_cavity, w1, w2, wc, wd, alpha1, alpha2, g, gamma, kappa,
        lambda_a=1.0, pulse=pulse, dissipation_model='non-Hermitian',
        gate=gate, iter_stop=1)


@pytest.mark.slowtest
def test_run_oct(oct_model, request, tmpdir):
    """Test the run_oct wrapper"""
    rf = str(tmpdir)
    oct_model.write_to_runfolder(rf)
    np.savetxt(
        os.path.join(rf, 'rwa_vector.dat'),
        oct_model.rwa_vector, header='rwa vector [MHz]')
    oct_model.gate.write(os.path.join(rf, 'target_gate.dat'), format='array')

    assert os.path.isfile(os.path.join(rf, 'target_gate.dat'))
    oct.run_oct(rf, scratch_root=tmpdir)
    with open(os.path.join(rf, 'oct.log')) as log_fh:
        print(log_fh.read())
    iters, J_T = np.genfromtxt(os.path.join(rf, "./oct_iters.dat"),
                               unpack=True, usecols=(0, 1))
    assert np.all(
        iters == np.array(
            [0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,
             1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  2.,  3.,  4.,  5.]))
    assert abs(J_T[-1] - 0.25858613671929997) < 1e-10


def test_filter_oct(oct_filter_model, request, tmpdir):
    """Test the run_oct wrapper"""
    rf = str(tmpdir)
    oct_filter_model.write_to_runfolder(rf)
    np.savetxt(
        os.path.join(rf, 'rwa_vector.dat'),
        oct_filter_model.rwa_vector, header='rwa vector [MHz]')
    oct_filter_model.gate.write(os.path.join(rf, 'target_gate.dat'),
                                format='array')

    def filter(freq):
        """Filter to ±200 MHz window. Relies on pulse freq_unit being MHz"""
        return np.abs(freq) < 200

    pulse = QDYN.pulse.Pulse.read(os.path.join(rf, 'pulse1.dat'))
    pulse.write_oct_spectral_filter(
        os.path.join(rf, 'filter.dat'), filter_func=filter, freq_unit='MHz')

    assert os.path.isfile(os.path.join(rf, 'target_gate.dat'))
    print("Runfolder: %s" % rf)
    oct.run_oct(rf, scratch_root=tmpdir)
    with open(os.path.join(rf, 'oct.log')) as log_fh:
        print(log_fh.read())
    J_T = np.genfromtxt(os.path.join(rf, "./oct_iters.dat"),
                        unpack=True, usecols=(1,))
    assert abs(J_T[-1] - 0.49560093373609998) < 1e-10
    print("Runfolder: %s" % rf)
