"""Tools and Wrappers for OCT"""
import os
import re
import uuid
import logging
import subprocess as sp
from glob import glob

import QDYN
from QDYN.pulse import Pulse


MAX_TRIALS = 200


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
        scratch_root=None):
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
            # TODO: the proper process
            oct_proc = sp.Popen(
                ['qdyn_optimize', '--internal-units=GHz_units.txt', '.'],
                cwd=temp_runfolder, env=env, stdout=sp.PIPE,
                universal_newlines=True)
            iteration = 0
            g_a_int = 0.0
            while True:  # monitor STDOUT from oct
                line = oct_proc.stdout.readline()
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
                        oct_proc.kill()
                        scale_lambda_a(temp_config, 0.5)
                        reset_pulse(temp_pulse_opt_dat, iteration)
                        break  # next bad_lambda loop
                    # if the pulse update explodes, we increase lambda_a (and
                    # prevent it from decreasing again)
                    if (('amplitude exceeds maximum value' in line) or
                            ('Loss of monotonic convergence' in line) or
                            (g_a_int > g_a_int_max)):
                        pulse_explosion = True
                        if "Loss of monotonic convergence" in line:
                            logger.debug("loss of monotonic conversion")
                        else:
                            if g_a_int > g_a_int_max:
                                logger.debug("g_a_int = %g > %g",
                                             g_a_int, g_a_int_max)
                            logger.debug("pulse explosion")
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
