#!/usr/bin/env python
import sys
from os.path import join

import numpy as np

import QDYN
from QDYN.shutil import mkdir
from QDYN.pulse import Pulse
from QDYN.gate2q import Gate2Q

from model import transmon_model
from oct import GATE

ORIG_ROOT = '/home/goerz/jobs/tmlandscape/runs_zeta_detailed/w2_5900MHz_wc_6200MHz'
NEW_ROOT = '/home/goerz/jobs/tmlandscape_revision/NODECAY'

FOLDERS = [
    '50ns_w_center_H_left', '50ns_w_center_H_right', '50ns_w_center_Ph_left',
    '50ns_w_center_Ph_right','PE_LI_BGATE_50ns_cont_SM']


def generate_runfolder(rf_orig, rf_new, gate):
    """Generate runfolder"""

    w1     = 6000.0 # MHz
    w2     = 5900.0 # MHz
    wc     = 6200.0 # MHz
    wd     = 5932.5 # MHz
    alpha1 = -290.0 # MHz
    alpha2 = -310.0 # MHz
    g      =   70.0 # MHz
    n_qubit = 5
    n_cavity = 6
    kappa = 0.0
    gamma = 0.0
    #kappa = list(np.arange(n_cavity) * 0.05)[1:-1] + [10000.0, ]  # MHz
    #gamma = [0.012, 0.024, 0.033, 10000.0]  # MHz

    pulse = Pulse.read(join(rf_orig, "pulse.dat"), freq_unit='MHz')
    pulse.config_attribs['is_complex'] = True
    #pulse.config_attribs['oct_spectral_filter'] = 'filter.dat'

    assert isinstance(gate, Gate2Q)
    model = transmon_model(
        n_qubit, n_cavity, w1, w2, wc, wd, alpha1, alpha2, g, gamma, kappa,
        lambda_a=1.0, pulse=pulse, dissipation_model='non-Hermitian',
        gate=gate, iter_stop=9000)
    model.write_to_runfolder(rf_new)

    def filter(freq):
        """Filter to ±200 MHz window. Relies on pulse freq_unit being MHz"""
        return np.abs(freq) < 200

    #pulse.write_oct_spectral_filter(join(rf_new, 'filter.dat'),
                                    #filter_func=filter, freq_unit='MHz')
    np.savetxt(join(rf_new, 'rwa_vector.dat'),
               model.rwa_vector, header='rwa vector [MHz]')
    gate.write(join(rf_new, 'target_gate.dat'), format='array')


def main(argv=None):
    for folder in FOLDERS:
        rf_orig = join(ORIG_ROOT, folder)
        rf_new = join(NEW_ROOT, folder)
        print("%s -> %s" % (rf_orig, rf_new))
        gate = Gate2Q.read(join(rf_orig, 'target_gate.dat'),
                           name='O', format='array')
        mkdir(rf_new)
        generate_runfolder(rf_orig, rf_new, gate)


if __name__ == "__main__":
    sys.exit(main())
