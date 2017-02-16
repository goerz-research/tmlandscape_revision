"""Tests for the model, ensuring that we can reproduce with earlier version of
QDYN and transmon programs. Run through `py.test`"""
import os
import sys
import subprocess

import pytest

import numpy as np
import sympy
import scipy.sparse
import qutip
import qnet.misc.testing_tools
from qnet.convert.to_qutip import convert_to_qutip
import QDYN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import model


datadir = pytest.fixture(qnet.misc.testing_tools.datadir)


@pytest.fixture
def H0_non_herm():
    """Non-Hermitian Symbolic Hamiltonian with non-linear decay"""
    return model.transmon_hamiltonian(n_qubit=5, n_cavity=6, non_herm=True,
                                      non_linear_decay=True)[0]


@pytest.fixture
def H0_num_non_herm(H0_non_herm):
    """Numerical non-Hermitian Hamiltonian, in MHz units"""
    δ1, δ2, Δ, α1, α2, g1, g2 = sympy.symbols(
        r'delta_1, delta_2, Delta, alpha_1, alpha_2, g_1, g_2', real=True)
    w1     = 6000.0  # MHz
    w2     = 5900.0  # MHz
    wc     = 6200.0  # MHz
    wd     = 5932.5  # MHz
    alpha1 = -290.0  # MHz
    alpha2 = -310.0  # MHz
    g      =   70.0  # MHz
    n_qubit = 5
    n_cavity = 6
    kappa = list(np.arange(n_cavity) * 0.05)[1:-1] + [10000.0, ]  # MHz
    gamma = [0.012, 0.024, 0.033, 10000.0]  # MHz
    num_vals = {δ1: w1 - wd, δ2: w2 - wd, Δ: wc - wd,
                α1: alpha1, α2: alpha2, g1: g, g2: g}
    for i in range(1, n_qubit):
        γ = sympy.symbols(r'gamma_%d' % i, real=True)
        num_vals[γ] = gamma[i-1]
    for n in range(1, n_cavity):
        κ = sympy.symbols(r'kappa_%d' % n, real=True)
        num_vals[κ] = kappa[n-1]
    return convert_to_qutip(H0_non_herm.substitute(num_vals),
                            full_space=H0_non_herm.space)


@pytest.fixture
def non_herm_model(datadir):
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
    pulse = QDYN.pulse.Pulse.read(os.path.join(datadir, "pulse.dat"))
    return model.transmon_model(
        n_qubit, n_cavity, w1, w2, wc, wd, alpha1, alpha2, g, gamma, kappa,
        lambda_a=0.93, pulse=pulse, dissipation_model='non-Hermitian')


@pytest.fixture
def bare_basis(H0_non_herm):
    """Return the qutip states that represent the bare two-qubit basis"""
    hs = H0_non_herm.space
    bare_00 = model.state(hs, 0, 0, 0, fmt='qutip')
    bare_01 = model.state(hs, 0, 1, 0, fmt='qutip')
    bare_10 = model.state(hs, 1, 0, 0, fmt='qutip')
    bare_11 = model.state(hs, 1, 1, 0, fmt='qutip')
    return [bare_00, bare_01, bare_10, bare_11]


def clip_array(a):
    filter = lambda v: v if abs(v) > 1e-4 else 0
    return np.array([filter(v) for v in a])


@pytest.fixture
def fortran_logical_states(datadir):
    """The logical states as they were identified by Fortran"""
    logical_states_dat = os.path.join(datadir, 'logical_states.dat')
    a00, a01, a10, a11 = np.genfromtxt(logical_states_dat, unpack=True,
                                       usecols=(0,1,2,3))
    ket00 = qutip.Qobj(clip_array(a00), dims=[[5, 5, 6], [1, 1, 1]])
    ket01 = qutip.Qobj(clip_array(a01), dims=[[5, 5, 6], [1, 1, 1]])
    ket10 = qutip.Qobj(clip_array(a10), dims=[[5, 5, 6], [1, 1, 1]])
    ket11 = qutip.Qobj(clip_array(a11), dims=[[5, 5, 6], [1, 1, 1]])
    return ket00, ket01, ket10, ket11


def same_state(state1, state2):
    """Check that two qutip states are the same"""
    return ((1.0 - abs(state1.overlap(state2))) < 1e-12)


def test_logical_basis(H0_num_non_herm, bare_basis, fortran_logical_states):
    """Test that Python identifies the same logical states as Fortran"""
    logical_basis = model.pick_logical_basis(H0_num_non_herm, bare_basis)
    assert same_state(logical_basis[0], fortran_logical_states[0])
    assert same_state(logical_basis[1], fortran_logical_states[1])
    assert same_state(logical_basis[2], fortran_logical_states[2])
    assert same_state(logical_basis[3], fortran_logical_states[3])


def test_drift_ham(H0_num_non_herm, datadir):
    """Test that drift Hamiltonian matches the one calculated in Fortran"""
    H0_num = H0_num_non_herm.data # in MHz
    H0_num_expected = QDYN.io.read_indexed_matrix(
        os.path.join(datadir, "ham_drift.dat")) * 1000.0  # GHz -> MHz
    H0_num_expected.eliminate_zeros()
    assert H0_num.nnz == H0_num_expected.nnz
    diff = H0_num - H0_num_expected
    rows, cols, vals = scipy.sparse.find(diff)
    for i, j, v in zip(rows, cols, vals):
        assert abs(v) < 1e-11


def test_prop_non_hermitian(non_herm_model, datadir, tmpdir):
    """Test that propagation of with a non-Hermitian Hamiltonian reproduces the
    same results as previous calculation"""
    rf = str(tmpdir)
    non_herm_model.write_to_runfolder(rf)
    np.savetxt(
        os.path.join(rf, 'rwa_vector.dat'),
        non_herm_model.rwa_vector, header='rwa vector [MHz]')
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    subprocess.check_call(
        ['qdyn_prop_gate', '--internal-units=GHz_units.txt', rf], env=env)
    U_expected = QDYN.gate2q.Gate2Q.read(os.path.join(datadir, "U.dat"),
                                         format='matrix')
    U_actual = list(
        QDYN.prop_gate.get_prop_gate_of_t(
            os.path.join(rf, 'U_over_t.dat'))
        )[-1]

    err = 1 - U_actual.closest_unitary().F_avg(U_expected.closest_unitary())
    assert err < 1e-10
