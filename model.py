"""Hamiltonian of the Transmon Model"""

import itertools

import sympy
import qutip
import numpy as np
from scipy.sparse import find as find_nonzero, coo_matrix

from qnet.algebra.hilbert_space_algebra import LocalSpace
from qnet.algebra.operator_algebra import Destroy, LocalSigma
from qnet.convert.to_qutip import convert_to_qutip

import QDYN
from QDYN.dissipation import lindblad_ops_to_dissipator
from QDYN.units import UnitFloat


def transmon_hamiltonian(n_qubit, n_cavity, non_herm=False,
        non_linear_decay=False):
    """Return the Transmon symbolic drift and control Hamiltonians. If
    `non_herm` is True, the drift Hamiltonian will include non-Hermitian decay
    terms for the spontaneous decay for the qubits and the cavity. This will be
    the "standard" decay with a linear decay rate, or with an independent decay
    rate for each level in the transmon and cavity if `non_linear_decay` is
    True.
    """
    HilQ1 = LocalSpace('1', dimension=n_qubit)
    HilQ2 = LocalSpace('2', dimension=n_qubit)
    HilCav = LocalSpace('c', dimension=n_cavity)
    b1 = Destroy(identifier='b_1', hs=HilQ1); b1_dag = b1.adjoint()
    b2 = Destroy(identifier='b_2', hs=HilQ2); b2_dag = b2.adjoint()
    a = Destroy(hs=HilCav); a_dag = a.adjoint()
    δ1, δ2, Δ, α1, α2, g1, g2 = sympy.symbols(
        r'delta_1, delta_2, Delta, alpha_1, alpha_2, g_1, g_2', real=True)
    H0 = (δ1 * b1_dag * b1 + (α1/2) * b1_dag * b1_dag * b1 * b1 +
          g1 * (b1_dag * a + b1 * a_dag) +
          δ2 * b2_dag * b2 + (α2/2) * b2_dag * b2_dag * b2 * b2 +
          g2 * (b2_dag * a + b2 * a_dag) +
          Δ * a_dag * a)
    if non_herm:
        if non_linear_decay:
            for i in range(1, n_qubit):
                γ = sympy.symbols(r'gamma_%d' % i, real=True)
                H0 = H0 - sympy.I * γ * LocalSigma(i, i, hs=HilQ1) / 2
            for j in range(1, n_qubit):
                γ = sympy.symbols(r'gamma_%d' % j, real=True)
                H0 = H0 - sympy.I * γ * LocalSigma(j, j, hs=HilQ2) / 2
            for n in range(1, n_cavity):
                κ = sympy.symbols(r'kappa_%d' % n, real=True)
                H0 = H0 - sympy.I * κ * LocalSigma(n, n, hs=HilCav) / 2
        else:
            γ, κ = sympy.symbols(r'gamma, kappa', real=True)
            H0 = H0 - sympy.I * γ * b1_dag * b1 / 2
            H0 = H0 - sympy.I * γ * b2_dag * b2 / 2
            H0 = H0 - sympy.I * κ * a_dag * a / 2
    H1 = a / 2  # factor 2 to account for RWA
    return H0, H1


def state(space, *numbers, fmt='qutip'):
    """Construct a state for a given QNET space by giving a quantum number for
    each sub-space

    Args:
    space (qnet.algebra.hilbert_space_algebra.HilbertSpace): The space in which
        the state lives
    numbers (tuple of ints): 0-based quantum numbers, one for each local factor
        in `space`
    fmt (str): output format. 'qutip' for a QuTiP state, 'numpy' for a numpy
        complex vector
    """
    states = []
    assert len(numbers) == len(space.local_factors)
    for i, hs in enumerate(space.local_factors):
        states.append(qutip.basis(hs.dimension, numbers[i]))
    if fmt == 'qutip':
        return qutip.tensor(*states)
    elif fmt == 'numpy':
        return QDYN.linalg.vectorize(qutip.tensor(*states).data.todense())
    else:
        raise ValueError("Unknown fmt")


def pick_logical_basis(H0_num, bare_basis):
    """Return the eigenstates of the real part of the QuTiP operator H0_num
    that define the logical basis 00, 01, 10, 11, as qutip states. The selected
    states are those that have the greatest overlap with `bare_basis`"""
    H0_num = qutip.Qobj(H0_num.data.real, dims=H0_num.dims)
    eigvecs = H0_num.eigenstates(sparse=False)[1]

    # normalize eigvecs to that real part of largest value is positive
    for (i_v, vec) in enumerate(eigvecs):
        val = find_nonzero(vec.data)[2]
        if val[np.argmax(np.abs(val))].real < 0:
            eigvecs[i_v] = -1 * eigvecs[i_v]

    # pick eigenstates that have the largest overlap with the bare states
    bare_00, bare_01, bare_10, bare_11 = bare_basis
    logical_levels = []

    logical_levels.append(  # 00
        np.argmax([abs(bare_00.overlap(v)) for v in eigvecs]))

    overlap01 = lambda i, v: 0 if i in logical_levels else bare_01.overlap(v)
    logical_levels.append(  # 01
        np.argmax([abs(overlap01(i, v)) for (i, v) in enumerate(eigvecs)]))

    overlap10 = lambda i, v: 0 if i in logical_levels else bare_10.overlap(v)
    logical_levels.append(  # 10
        np.argmax([abs(overlap10(i, v)) for (i, v) in enumerate(eigvecs)]))

    overlap11 = lambda i, v: 0 if i in logical_levels else bare_11.overlap(v)
    logical_levels.append(  # 11
        np.argmax([abs(overlap11(i, v)) for (i, v) in enumerate(eigvecs)]))

    return tuple([eigvecs[i] for i in logical_levels])



def transmon_model(n_qubit, n_cavity, w1, w2, wc, wd, alpha1, alpha2, g,
        gamma, kappa, lambda_a, pulse, dissipation_model='dissipator'):
    """Return a QDYN model for propagation of a 2-transmon system

    Args:
        n_qubit (int): number of levels after which transmon is trunctaed
        n_cavity (int): number of levels after which cavity is trunctaed
        w1 (float): frequency of qubit 1 (MHz)
        w2 (float): frequency of qubit 2 (MHz)
        wc (float): frequency of cavity (MHz)
        wd (float): frequency of rotating frame (MHz)
        alpha1 (float): anharmonicity of transmon 1 (MHz)
        alpha2 (float): anharmonicity of transmon 2 (MHz)
        g (float): transmon-cavity coupling (MHz)
        gamma (float): decay rate of transmon (MHz). May be a list of values of
            size ``n_qubit - 1`` for non-linear decay, for
            ``dissipation_mode='non_Hermitian' only``
        kappa (float): decay rate of cavity (MHz). May be a list of value of
            size ``n_cavity - 1`` for non-linear decay, for
            ``dissipation_mode='non_Hermitian' only``. Note that `gamma` and
            `kappa` must either both be floats or both be lists.
        lambda_a (float): Krotov scaling parameter
        pulse (QDYN.pulse.Pulse): control pulse
        dissipation_model (str): one of 'dissipator' (density matrix
            propagation), 'non-Hermitian' (Hilbert space propagation with
            non-Hermitian Hamiltonian)

    The returned model has an `rwa_vector` custom attribute.
    """
    δ1, δ2, Δ, α1, α2, g1, g2 = sympy.symbols(
        r'delta_1, delta_2, Delta, alpha_1, alpha_2, g_1, g_2', real=True)
    num_vals = {  # numeric values in the RWA
            δ1: w1 - wd, δ2: w2 - wd, Δ: wc - wd,
            α1: alpha1, α2: alpha2, g1: g, g2: g}
    nt = len(pulse.tgrid) + 1
    t0 = pulse.t0
    T = pulse.T
    if dissipation_model == 'non-Hermitian':
        non_herm = True
        assert isinstance(gamma, type(kappa))
        if isinstance(gamma, float):
            non_linear_decay = False
            γ, κ = sympy.symbols('gamma, kappa', real=True)
            num_vals[γ] = gamma
            num_vals[κ] = kappa
        else:
            non_linear_decay = True
            assert len(gamma) == n_qubit - 1
            assert len(kappa) == n_cavity - 1
            for i in range(1, n_qubit):
                γ = sympy.symbols(r'gamma_%d' % i, real=True)
                num_vals[γ] = gamma[i-1]
            for n in range(1, n_cavity):
                κ = sympy.symbols(r'kappa_%d' % n, real=True)
                num_vals[κ] = kappa[n-1]
    else:
        non_herm = False
        non_linear_decay = False
        # non-linear decay rates are not allowed
        assert isinstance(gamma, float)
        assert isinstance(kappa, float)

    H0, H1 = transmon_hamiltonian(n_qubit, n_cavity, non_herm=non_herm,
                                  non_linear_decay=non_linear_decay)
    hs = H0.space
    H0_num = convert_to_qutip(H0.substitute(num_vals), full_space=hs)
    H1_num = convert_to_qutip(H1, full_space=H0.space)

    model = QDYN.model.LevelModel()

    # dissipation model
    if dissipation_model == 'dissipator':
        # Use a dissipator (density matrix propagation); linear decay
        decay_ops = {ls.label: Destroy(hs=ls) for ls in H0.space.local_factors}
        L1 = np.sqrt(gamma) * convert_to_qutip(decay_ops['1'], full_space=hs)
        L2 = np.sqrt(gamma) * convert_to_qutip(decay_ops['2'], full_space=hs)
        Lc = np.sqrt(kappa) * convert_to_qutip(decay_ops['c'], full_space=hs)
        lindblad_ops = [L1, L2, Lc]
        D = lindblad_ops_to_dissipator(
                [coo_matrix(L.data) for L in lindblad_ops])
        model.set_dissipator(D, op_unit='MHz')
    elif dissipation_model == 'non-Hermitian':
        # the decay term is already included in H0 and H0_num
        pass
    else:
        raise ValueError("Unknown dissipatoin_model: %s" % dissipation_model)

    # Hamiltonian
    model.add_ham(H0_num, op_unit='MHz', op_type='potential')
    model.add_ham(H1_num, pulse=pulse, op_unit='dimensionless',
                  op_type='dipole')
    model.add_ham(H1_num.dag(), pulse=pulse, op_unit='dimensionless',
                  op_type='dipole', conjg_pulse=True)

    model.set_propagation(T=T, nt=nt, t0=t0, time_unit='ns',
                          prop_method='newton')

    # RWA vector
    model.rwa_vector = wd * np.array(
            [sum(ijn) for ijn in itertools.product(
                *[list(range(n)) for n in H0_num.dims[0]])])

    # States
    bare_00 = state(H0.space, 0, 0, 0, fmt='qutip')
    bare_01 = state(H0.space, 0, 1, 0, fmt='qutip')
    bare_10 = state(H0.space, 1, 0, 0, fmt='qutip')
    bare_11 = state(H0.space, 1, 1, 0, fmt='qutip')
    bare_basis = [bare_00, bare_01, bare_10, bare_11]
    dressed_basis = pick_logical_basis(H0_num, bare_basis)
    dressed_00, dressed_01, dressed_10, dressed_11 = dressed_basis
    model.add_state(dressed_00, label='00')
    model.add_state(dressed_01, label='01')
    model.add_state(dressed_10, label='10')
    model.add_state(dressed_11, label='11')

    # OCT
    pulse_settings = {
        pulse: {
            'oct_outfile': 'pulse.oct.dat',
            'oct_lambda_a': lambda_a, 'oct_lambda_intens': 0.0,
            'oct_increase_factor': 5.0, 'oct_shape': 'flattop',
            'shape_t_start': 0.0, 't_rise': UnitFloat(2.0, 'ns'),
            'shape_t_stop': T, 't_fall': UnitFloat(2.0, 'ns'),
            }
    }
    model.set_oct(pulse_settings, method='krotovpk', J_T_conv=1e01,
                  max_ram_mb=8000)

    model.user_data['time_unit'] = 'ns'
    model.user_data['rwa_vector'] = 'rwa_vector.dat'
    model.user_data['write_gate'] = 'U_over_t.dat'
    model.user_data['basis'] = '00,01,10,11'
    return model
