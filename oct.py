"""Tools and Wrappers for OCT"""
import os
import re
import uuid
import logging
import subprocess as sp
from glob import glob
from shutil import rmtree, copytree
from scipy.optimize import minimize
from QDYN.analytical_pulse import AnalyticalPulse
from math import atanh, tanh

import numpy as np

import QDYN
from QDYN.pulse import Pulse
from QDYN.analytical_pulse import AnalyticalPulse
from QDYN.prop_gate import get_prop_gate_of_t

from model import transmon_model


HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
PHASE = np.array([[1, 0], [0, np.exp(-0.25j*np.pi)]], dtype=np.complex128)

# gate targets
GATE = {
    'H_left'  : QDYN.gate2q.Gate2Q(np.kron(HADAMARD, np.eye(2)), name='H_L'),
    'H_right' : QDYN.gate2q.Gate2Q(np.kron(np.eye(2), HADAMARD), name='H_R'),
    'Ph_left' : QDYN.gate2q.Gate2Q(np.kron(PHASE, np.eye(2)), name='S_L'),
    'Ph_right': QDYN.gate2q.Gate2Q(np.kron(np.eye(2), PHASE), name='S_R'),
    'BGATE'   : QDYN.gate2q.BGATE,
}


MAX_TRIALS = 200


w1     = 6000.0  # MHz
w2     = 5900.0  # MHz
wc     = 6200.0  # MHz
alpha1 = -290.0  # MHz
alpha2 = -310.0  # MHz
g      =   70.0  # MHz


def reset_pulse(pulse_dat, iteration):
    """Reset pulse_dat at the given iteration to the last available snapshot,
    assuming that snapshots are available using the same name as pulse_dat,
    with the `iteration` append to the file name (e.g. pulse.dat.100 for
    pulse.dat).  Snapshots must be at least 10 iterations older than the
    current pulse"""
    snapshot_list = glob("%s.*" % pulse_dat)
    snapshots = {}
    logger = logging.getLogger(__name__)
    logger.debug("resetting in iter %d", iteration)
    logger.debug("available snapshots: %s", str(snapshot_list))
    for snapshot in snapshot_list:
        try:
            snapshot_iter = int(os.path.splitext(snapshot)[1][1:])
            snapshots[snapshot_iter] = snapshot
        except ValueError:
            pass  # ignore pulse.dat.prev
    snapshot_iters = sorted(snapshots.keys())
    os.unlink(pulse_dat)
    while len(snapshot_iters) > 0:
        snapshot_iter = snapshot_iters.pop()
        if (iteration == 0) or (snapshot_iter + 10 < iteration):
            logger.debug("accepted snapshot: %s (iter %d)",
                         snapshots[snapshot_iter], snapshot_iter)
            QDYN.shutil.copy(snapshots[snapshot_iter], pulse_dat)
            return
        else:
            logger.debug("rejected snapshot: %s (iter %d)",
                         snapshots[snapshot_iter], snapshot_iter)
    logger.debug("no accepted snapshot")


def get_temp_runfolder(runfolder, scratch_root=None):
    """Return the path for an appropriate temporary runfolder (inside
    $SCRATCH_ROOT) for the given "real" runfolder. The runfolder is guaranteed
    to exist"""
    if scratch_root is None:
        assert 'SCRATCH_ROOT' in os.environ, \
            "SCRATCH_ROOT environment variable must be defined"
        scratch_root = os.environ['SCRATCH_ROOT']
    temp_runfolder = str(uuid.uuid4())
    if 'SLURM_JOB_ID' in os.environ:
        temp_runfolder = "%s_%s" % (os.environ['SLURM_JOB_ID'], temp_runfolder)
    temp_runfolder = os.path.join(scratch_root, runfolder, temp_runfolder)
    QDYN.shutil.mkdir(temp_runfolder)
    return temp_runfolder


def run_oct(
        runfolder, continue_oct=False, g_a_int_min_initial=1.0e-5,
        g_a_int_max=1.0e-1, g_a_int_converged=1.0e-7, use_threads=True,
        scratch_root=None, print_stdout=True, monotonic=True, backtrack=True):
    """Run optimal control on the given runfolder. Adjust lambda_a if
    necessary.

    Assumes that the runfolder contains the files config and pulse1.dat, and
    optionally, pulse.dat, and oct_iters.dat.

    Creates (overwrites) the files pulse.dat and oct_iters.dat.

    Also, a file config.oct is created that contains the last update to
    lambda_a. The original config file will remain unchanged.
    """
    logger = logging.getLogger(__name__)

    config_data = QDYN.config.read_config_file(
                        os.path.join(runfolder, 'config'))
    assert len(config_data['pulse'])

    # prepare clean temp_runfolder
    pulse_guess_dat = config_data['pulse'][0]['filename']
    pulse_opt_dat = config_data['pulse'][0]['oct_outfile']
    rf_pulse_guess_dat = os.path.join(runfolder, pulse_guess_dat)
    rf_pulse_opt_dat = os.path.join(runfolder, pulse_opt_dat)
    temp_runfolder = get_temp_runfolder(runfolder, scratch_root)
    temp_pulse_opt_dat = os.path.join(temp_runfolder, pulse_opt_dat)
    temp_config = os.path.join(temp_runfolder, 'config')
    assert 'basis' in config_data['user_strings']
    assert 'J_T' in config_data['user_strings']
    assert 'gate' in config_data['user_strings']
    QDYN.config.write_config(config_data, temp_config)
    if pulse_is_converged(rf_pulse_opt_dat):
        logger.warning("pulse %s is already converged. Skipping.",
                       rf_pulse_opt_dat)
        return
    ham_files = [line.get('filename', None) for line in config_data['ham']]
    psi_files = [line.get('filename', None) for line in config_data['psi']]
    pulse_files = [line.get('filename', None) for line in config_data['pulse']]
    pulse_files.extend([line.get('oct_spectral_filter', None)
                        for line in config_data['pulse']])
    user_files = [config_data['user_strings'].get(key, None)
                  for key in ['rwa_vector', 'gate']]
    required_files = [fn for fn in (
                        ham_files + psi_files + pulse_files + user_files)
                      if fn is not None]
    files_to_copy = list(required_files)
    if continue_oct:
        files_to_copy.extend([pulse_opt_dat, 'oct_iters.dat'])
    for file in files_to_copy:
        if os.path.isfile(os.path.join(runfolder, file)):
            QDYN.shutil.copy(os.path.join(runfolder, file), temp_runfolder)
            logger.debug("%s to temp_runfolder %s", file, temp_runfolder)
        else:
            if file in required_files:
                raise IOError("%s does not exist in %s" % (file, runfolder))
    logger.info("Starting optimization of %s (in %s)", runfolder,
                temp_runfolder)

    # run while monitoring convergence
    with open(os.path.join(runfolder, 'oct.log'), 'wb', 0) as stdout:
        # we assume that the value for lambda_a is badly chosen and iterate
        # over optimizations until we find a good value
        bad_lambda = True
        pulse_explosion = False
        trial = 0
        given_up = False
        while bad_lambda:
            trial += 1
            if trial > MAX_TRIALS:
                bad_lambda = False
                given_up = True
                break  # give up
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '1'
            if use_threads:
                env['OMP_NUM_THREADS'] = '4'
                if int(use_threads) > 1:
                    env['OMP_NUM_THREADS'] = '%d' % int(use_threads)
            oct_proc = sp.Popen(
                ['qdyn_optimize', '--internal-units=GHz_units.txt', '.'],
                cwd=temp_runfolder, env=env, stdout=sp.PIPE,
                universal_newlines=True)
            iteration = 0
            g_a_int = 0.0
            while True:  # monitor STDOUT from oct
                line = oct_proc.stdout.readline()
                if print_stdout:
                    print(line, end='')
                if line != '':
                    stdout.write(line.encode('ascii'))
                    m = re.search(r'^\s*(\d+) \| [\d.E+-]+ \| ([\d.E+-]+) \|',
                                  line)
                    if m:
                        iteration = int(m.group(1))
                        try:
                            g_a_int = float(m.group(2))
                        except ValueError:
                            # account for Fortran dropping the 'E' in negative
                            # 3-digit exponents
                            g_a_int = float(m.group(2).replace('-', 'E-'))
                    # Every 50 iterations, we take a snapshot of the current
                    # pulse, so that "bad lambda" restarts continue from there
                    if (iteration > 0) and (iteration % 50 == 0):
                        QDYN.shutil.copy(temp_pulse_opt_dat,
                                         temp_pulse_opt_dat+'.'+str(iteration))
                    # if the pulse changes in first iteration are too small, we
                    # lower lambda_a, unless lambda_a was previously adjusted
                    # to avoid exploding pulse values
                    if ((iteration == 1) and
                            (g_a_int < g_a_int_min_initial) and
                            (not pulse_explosion)):
                        logger.debug("pulse update too small: %g < %g",
                                     g_a_int, g_a_int_min_initial)
                        logger.debug("Kill %d", oct_proc.pid)
                        if backtrack:
                            oct_proc.kill()
                            scale_lambda_a(temp_config, 0.5)
                            reset_pulse(temp_pulse_opt_dat, iteration)
                            break  # next bad_lambda loop
                    # if the pulse update explodes, we increase lambda_a (and
                    # prevent it from decreasing again)
                    need_to_increase_lambda = False
                    if 'amplitude exceeds maximum value' in line:
                        need_to_increase_lambda = True
                    if monotonic:
                        if 'Loss of monotonic convergence' in line:
                            need_to_increase_lambda = True
                    if g_a_int > g_a_int_max:
                        need_to_increase_lambda = True
                    if need_to_increase_lambda:
                        pulse_explosion = True
                        if "Loss of monotonic convergence" in line:
                            logger.debug("loss of monotonic conversion")
                        else:
                            if g_a_int > g_a_int_max:
                                logger.debug("g_a_int = %g > %g",
                                             g_a_int, g_a_int_max)
                            logger.debug("pulse explosion")
                        if backtrack:
                            logger.debug("Kill %d", oct_proc.pid)
                            oct_proc.kill()
                            scale_lambda_a(temp_config, 1.25)
                            reset_pulse(temp_pulse_opt_dat, iteration)
                            break  # next bad_lambda loop
                    # if there are no significant pulse changes anymore, we
                    # stop the optimization prematurely
                    if iteration > 10 and g_a_int < g_a_int_converged:
                        logger.debug(
                            ("pulse update insignificant "
                             "(converged): g_a_int = %g < %g"),
                            g_a_int, g_a_int_converged)
                        logger.debug("Kill %d", oct_proc.pid)
                        oct_proc.kill()
                        bad_lambda = False
                        # add a comment to pulse.dat to mark it converged
                        mark_pulse_converged(temp_pulse_opt_dat)
                        break  # effectively break from bad_lambda loop
                else:  # line == ''
                    # OCT finished
                    bad_lambda = False
                    break  # effectively break from bad_lambda loop
    for file in [pulse_opt_dat, 'oct_iters.dat']:
        if os.path.isfile(os.path.join(temp_runfolder, file)):
            QDYN.shutil.copy(os.path.join(temp_runfolder, file), runfolder)
    if os.path.isfile(temp_config):
        QDYN.shutil.copy(temp_config, os.path.join(runfolder, 'config.oct'))
    QDYN.shutil.rmtree(temp_runfolder)
    logger.debug("Removed temp_runfolder %s", temp_runfolder)
    if given_up:
        # Giving up is permanent, so we can mark the guess pulse as final
        # by storing it as the optimized pulse. That should prevent pointlessly
        # re-runing OCT
        if not os.path.isfile(rf_pulse_opt_dat):
            QDYN.shutil.copy(rf_pulse_guess_dat, rf_pulse_opt_dat)
            mark_pulse_converged(rf_pulse_opt_dat)
        logger.info("Finished optimization (given up after too many "
                    "attempts): %s", runfolder)
    else:
        logger.info("Finished optimization: %s", runfolder)


def scale_lambda_a(config_file, factor):
    """Scale lambda_a in the given config file with the given factor"""
    config = QDYN.config.read_config_file(config_file)
    assert len(config['pulse']) == 1
    config['pulse'][0]['oct_lambda_a'] = float(
        "%.2e" % (factor * config['pulse'][0]['oct_lambda_a']))
    QDYN.config.write_config(config, config_file)


def mark_pulse_converged(pulse_file):
    """Mark pulse file a converged by writing a comment to the header"""
    if not pulse_is_converged(pulse_file):
        p = Pulse.read(pulse_file)
        p.preamble.append("# converged")
        p.write(pulse_file)


def pulse_is_converged(pulse_file):
    """Check if pulse file has 'converged' mark in header"""
    if not os.path.isfile(pulse_file):
        return False
    p = Pulse.read(pulse_file)
    for line in p.preamble:
        if 'converged' in line:
            return True
    return False


def blackman100ns(tgrid, E0):
    from QDYN.pulse import blackman
    assert (tgrid[-1] + tgrid[0] - 100) < 1e-10
    T =  100
    return E0 * blackman(tgrid, 0, T)


AnalyticalPulse.register_formula('blackman100ns', blackman100ns)


def get_U(pulse, wd, gate=None, J_T=None, dissipation=True,
        keep_runfolder=None):
    """Propagate pulse in the given rotating frame, using the non-Hermitian
    Schrödinger equation, and return the resulting (non-unitary, due to
    population loss) gate U"""

    assert 5000 < wd < 7000
    assert isinstance(pulse, QDYN.pulse.Pulse)
    rf = get_temp_runfolder('evaluate_universal_hs')
    n_qubit = 5
    n_cavity = 6
    kappa = list(np.arange(n_cavity) * 0.05)[1:-1] + [10000.0, ]  # MHz
    gamma = [0.012, 0.024, 0.033, 10000.0]  # MHz
    if not dissipation:
        kappa = list(np.arange(n_cavity) * 0.0)[1:-1] + [0.0, ]  # MHz
        gamma = [0.0, 0.0, 0.0, 0.0]  # MHz

    if gate is None:
        gate = GATE['BGATE']
    assert isinstance(gate, QDYN.gate2q.Gate2Q)
    if J_T is None:
        J_T = 'sm'

    model = transmon_model(
        n_qubit, n_cavity, w1, w2, wc, wd, alpha1, alpha2, g, gamma, kappa,
        lambda_a=1.0, pulse=pulse, dissipation_model='non-Hermitian',
        gate=gate, J_T=J_T, iter_stop=1)

    # write to runfolder
    model.write_to_runfolder(rf)
    np.savetxt(
        os.path.join(rf, 'rwa_vector.dat'),
        model.rwa_vector, header='rwa vector [MHz]')
    gate.write(os.path.join(rf, 'target_gate.dat'), format='array')

    # propagate
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '4'
    try:
        stdout = sp.check_output(
            ['qdyn_prop_gate', '--internal-units=GHz_units.txt', rf], env=env,
            universal_newlines=True)
    except sp.CalledProcessError as exc_info:
        from IPython.core.debugger import Tracer
        Tracer()()
        print(exc_info)

    # evaluate error
    for U_t in get_prop_gate_of_t(os.path.join(rf, 'U_over_t.dat')):
        U = U_t
    if keep_runfolder is not None:
        if os.path.isdir(keep_runfolder):
            rmtree(keep_runfolder)
        copytree(rf, keep_runfolder)
    rmtree(rf)

    return U


def evaluate_pulse_rho(pulse, gate, wd, n_qubit=5, n_cavity=6, silent=False):
    """Propagate pulse in Liouville space"""
    n_qubit = n_qubit
    n_cavity = n_cavity
    kappa = 0.05 # MHz
    gamma = 0.012 # MHz

    rf = get_temp_runfolder('evaluate_universal_rho')

    if isinstance(gate, str):
        gate = GATE[gate]
    assert isinstance(gate, QDYN.gate2q.Gate2Q)

    if not silent:
        print("preprocessing in %s" % rf)
    model = transmon_model(
        n_qubit, n_cavity, w1, w2, wc, wd, alpha1, alpha2, g, gamma, kappa,
        lambda_a=1.0, pulse=pulse, dissipation_model='dissipator',
        gate=gate)

    # write to runfolder
    model.write_to_runfolder(rf)
    np.savetxt(
        os.path.join(rf, 'rwa_vector.dat'),
        model.rwa_vector, header='rwa vector [MHz]')
    gate.write(os.path.join(rf, 'target_gate.dat'), format='array')

    # propagate
    if not silent:
        print("starting propagation in %s" % rf)
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '16'
    try:
        stdout = sp.check_output(
            ['qdyn_prop_gate', '--rho', '--internal-units=GHz_units.txt', rf],
            env=env, universal_newlines=True)
    except sp.CalledProcessError as exc_info:
        from IPython.core.debugger import Tracer
        Tracer()()
        print(exc_info)
    err = float(re.search(r'1-F_avg\(U, O\)\s*=\s*([Ee.0-9+-]*)',
                          stdout).group(1))
    if not silent:
        print("err_avg = %.4e" % err)
    return err


def evaluate_pulse(pulse, gate, wd, dissipation=True):
    """Evaluate figure of merit for how well the pulse implements the given
    gate (for simplex). For local gates, the figure of merit is 1-Favg, for
    BGATE it is J_T_LI + population loss"""

    # calculate model
    if isinstance(gate, QDYN.gate2q.Gate2Q):
        O = gate
        gate = 'O'
    else:
        O = GATE[gate]
    J_T = 'sm'
    if gate == 'BGATE':
        J_T = 'LI'
    U = get_U(pulse, wd, gate=O, J_T=J_T, dissipation=dissipation)
    err = 1-U.F_avg(O)
    if gate == 'BGATE':
        err = QDYN.weyl.J_T_LI(O, U) + U.pop_loss()

    return err


def krotov_from_pulse(
        gate, wd, pulse, iter_stop=100, dissipation=True,
        ens_pulse_scale=None, freq_window=200, lambda_a=1.0,
        g_a_int_converged=1.0e-7):
    """Run a Krotov optimization from the given guess pulse"""
    n_qubit = 5
    n_cavity = 6
    kappa = list(np.arange(n_cavity) * 0.05)[1:-1] + [10000.0, ]  # MHz
    gamma = [0.012, 0.024, 0.033, 10000.0]  # MHz
    if not dissipation:
        kappa = list(np.arange(n_cavity) * 0.0)[1:-1] + [10000.0, ]  # MHz
        gamma = [0.0, 0.0, 0.0, 10000.0]  # MHz

    assert 5000 < wd < 7000
    assert isinstance(pulse, QDYN.pulse.Pulse)

    pulse.config_attribs['is_complex'] = True
    if freq_window is not None:
        pulse.config_attribs['oct_spectral_filter'] = 'filter.dat'

    if isinstance(gate, QDYN.gate2q.Gate2Q):
        rf = get_temp_runfolder('krotov_O')
        O = gate
        gate = 'O'
    else:
        rf = get_temp_runfolder('krotov_%s' % gate)
        O = GATE[gate]

    J_T = 'sm'
    if gate == 'BGATE':
        J_T = 'LI'

    use_threads = True
    if ens_pulse_scale is not None:
        use_threads = (len(ens_pulse_scale) + 1) * 4
    model = transmon_model(
        n_qubit, n_cavity, w1, w2, wc, wd, alpha1, alpha2, g, gamma, kappa,
        lambda_a=lambda_a, pulse=pulse, dissipation_model='non-Hermitian',
        gate=O, iter_stop=iter_stop, J_T=J_T, ens_pulse_scale=ens_pulse_scale)
    model.write_to_runfolder(rf)
    np.savetxt(
        os.path.join(rf, 'rwa_vector.dat'),
        model.rwa_vector, header='rwa vector [MHz]')
    O.write(os.path.join(rf, 'target_gate.dat'), format='array')

    def filter(freq):
        """Filter to ± `freq_window` MHz window."""
        return np.abs(freq) < freq_window

    if freq_window is not None:
        pulse.write_oct_spectral_filter(
            os.path.join(rf, 'filter.dat'), filter_func=filter,
            freq_unit='MHz')
    print("Runfolder: %s" % rf)
    run_oct(rf, scratch_root=rf, monotonic=False, use_threads=use_threads,
            g_a_int_converged=g_a_int_converged)
    print("Runfolder: %s" % rf)
    opt_pulse = Pulse.read(os.path.join(rf, "pulse.oct.dat"))
    err = evaluate_pulse(opt_pulse, O, wd, dissipation=dissipation)
    print("1-F_avg = %.5e" % err)
    return opt_pulse


def u_tanh(v, v_min, v_max):
    return atanh((2 * v - (v_max + v_min)) / (v_max - v_min))


def v_tanh(u, v_min, v_max):
    return 0.5 * (v_max - v_min) * tanh(u) + 0.5 * (v_max + v_min)


def simplex_wd_E0(gate, x0, T=100):

    def fun(x):
        u_wd, u_E0 = x
        wd = v_tanh(u_wd, 5830, 6035)
        E0 = v_tanh(u_E0, 10, 300)
        pulse = AnalyticalPulse(
                "blackman100ns", T=T, nt=2000, time_unit='ns',
                ampl_unit='MHz', parameters={'E0': E0}).to_num_pulse()
        f = evaluate_pulse(pulse, gate, wd)
        print("E0 = %.2f\tw_d = %.2f:\t%.4e" % (E0, wd, f))
        return f

    wd_0, E0_0 = x0
    u_wd_0 = u_tanh(wd_0, 5830, 6035)
    u_E0_0 = u_tanh(E0_0, 10, 300)

    res = minimize(fun, (u_wd_0, u_E0_0), method='Nelder-Mead')
    return res.x


def simplex_E0(gate, wd, E0_0, T=100):

    def fun(x):
        u_E0 = x[0]
        E0 = v_tanh(u_E0, 10, 300)
        pulse = AnalyticalPulse(
                "blackman100ns", T=T, nt=2000, time_unit='ns',
                ampl_unit='MHz', parameters={'E0': E0}).to_num_pulse()
        f = evaluate_pulse(pulse, gate, wd)
        print("E0 = %.2f\tw_d = %.2f:\t%.4e" % (E0, wd, f))
        return f

    u_E0_0 = u_tanh(E0_0, 10, 300)

    res = minimize(fun, (u_E0_0, ), method='Nelder-Mead')
    return res.x


def krotov_from_blackman(gate, wd, E0, T=100, iter_stop=100, dissipation=True,
        ens_pulse_scale=None, freq_window=200):
    from oct import krotov_from_pulse
    pulse = AnalyticalPulse(
            "blackman100ns", T=T, nt=2000, time_unit='ns',
            ampl_unit='MHz', parameters={'E0': E0}).to_num_pulse()
    return krotov_from_pulse(gate, wd, pulse, iter_stop=iter_stop,
                             dissipation=dissipation,
                             ens_pulse_scale=ens_pulse_scale,
                             freq_window=freq_window)
