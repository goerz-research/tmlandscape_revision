#!/usr/bin/env python
import os
from os.path import join
import sys
import QDYN
from QDYN.prop_gate import get_prop_gate_of_t
import numpy as np
from numpy import sin, cos, pi, angle
import matplotlib
from collections import OrderedDict, defaultdict
matplotlib.use('Agg')
import matplotlib.pylab as plt
from mgplottools.mpl import new_figure, set_axis, get_color, ls
import pandas as pd

"""
Plot population / pulse dynamics
"""

STYLE = 'paper.mplstyle'

OUTFOLDER = './paper_images'
#OUTFOLDER = '/Users/goerz/Documents/Papers/TransmonLandscape'


def render_eqnarray(
        fig, fig_width, fig_height, x, y, eq_gap, lineheight, eqns):
    """Render an array of equations onto `fig`"""
    for i, (lhs, rhs) in enumerate(eqns):
        fig.text((x - eq_gap)/fig_width, (y - i * lineheight)/fig_height,
                 "$%s$" % lhs, verticalalignment='center',
                 horizontalalignment='right')
        fig.text((x)/fig_width, (y - i * lineheight)/fig_height,
                 '$=$', verticalalignment='center',
                 horizontalalignment='center')
        fig.text((x + eq_gap)/fig_width, (y - i * lineheight)/fig_height,
                 "$%s$" % rhs, verticalalignment='center',
                 horizontalalignment='left')


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def render_universal_pulse_plot(universal_rf, field_free_rf, errors_diss,
        errors_nodiss, wd, axes_info):
    fig_width    = 18.0
    fig_height   = 13.2
    spec_offset  =  0.7
    phase_deriv_offset =  2.95
    pulse_offset = 4.45
    phase_deriv_h =  1.5
    label_offset = 13.1
    error_offset = 12.5
    spec_h       =  1.5
    pulse_h      =  1.5
    left_margin  =  1.4
    right_margin =  0.25
    gap          =  0.0 # horizontal gap between panels
    y_label_offset  = 0.07
    log_offset = 6.8
    log_h = 1.0
    dyn_offset = 8.5
    dyn_width = 1.55

    fig = new_figure(fig_width, fig_height, style=STYLE)

    w = float(fig_width - (left_margin + right_margin + 4 * gap)) / 5

    labels = {
            'H_L': r'Hadamard (1)',
            'H_R': r'Hadamard (2)',
            'S_L': r'Phasegate (1)',
            'S_R': r'Phasegate (2)',
            'PE': r'BGATE',
    }

    w_center = 5932.5

    polar_axes = []

    for i_tgt, tgt in enumerate(['H_L', 'H_R', 'S_L', 'S_R', 'PE']):

        left_offset = left_margin + i_tgt * (w+gap)

        p = QDYN.pulse.Pulse.read(
                os.path.join(universal_rf[tgt], 'pulse1.dat'), freq_unit='MHz')
        err_diss = errors_diss[tgt]  #  Liouville space error
        err_nodiss = errors_nodiss[tgt]  #  Hilbert space error
        freq, spectrum = p.spectrum(mode='abs', sort=True)
        spectrum *= 1.0 / len(spectrum)

        # column labels, errors
        fig.text((left_offset + 0.5*w)/fig_width, label_offset/fig_height,
                 labels[tgt], verticalalignment='top',
                 horizontalalignment='center')
        eqns = (
            (r'\varepsilon^{\text{no diss.}}_{\text{avg}}',
                latex_exp(err_nodiss)),
            (r'\varepsilon^{\text{diss.}}_{\text{avg}}',
                latex_exp(err_diss)))
        render_eqnarray(fig, fig_width, fig_height, left_offset + 0.5*w - 0.3,
                error_offset, eq_gap=0.15, lineheight=0.5, eqns=eqns)

        # spectrum
        pos = [left_offset/fig_width, spec_offset/fig_height,
               w/fig_width, spec_h/fig_height]
        ax_spec = fig.add_axes(pos)
        ax_spec.plot(freq, 1.1*spectrum, label='spectrum')
        a = axes_info['spec']['x']
        set_axis(ax_spec, 'x', a.start, a.stop, range=a.range, step=a.step,
                 minor=a.minor, label=r'$\Delta f$ (MHz)', labelpad=1)
        offset = w_center - wd[tgt]
        delta1 = offset + 0.5 * (49.82 + 50.11) # MHz
        delta2 = offset + 0.5 * (-50.25 -49.95) # MHz
        alpha1 = 0.5 * (-225.65 -219.73) # MHz
        alpha2 = 0.5 * (-341.00 -347.32 ) # MHz
        # Note: the above frequencies are "dressed", cf SpectralAnalysis.ipynb
        # The the splitting due to the static interaction (i.e., "other qubit
        # in 0 or 1") is small, so we just average the two values and draw a
        # single line
        ax_spec.axvline(x=delta2, ls='--', color=get_color('green'))
        ax_spec.axvline(x=delta1, ls='--', color=get_color('orange'))
        ax_spec.text(x=(delta2-20), y=85, s=r'2',
                     ha='right', va='top', color=get_color('green'),
                     bbox=dict(linewidth=0, facecolor='white', alpha=0.0))
        ax_spec.text(x=(delta1+20), y=85, s=r'1',
                     ha='left', va='top', color=get_color('orange'),
                     bbox=dict(linewidth=0, facecolor='white', alpha=0.0))
        ax_spec.axvline(x=(delta2+alpha2), ls='dotted',
                        color=get_color('green'))
        ax_spec.axvline(x=(delta1+alpha1), ls='dotted',
                        color=get_color('orange'))
        a = axes_info['spec']['y']
        if i_tgt == 0:
            set_axis(ax_spec, 'y', a.start, a.stop, range=a.range, step=a.step,
                     minor=a.minor, label='')
        else:
            set_axis(ax_spec, 'y', a.start, a.stop, range=a.range, step=a.step,
                     minor=a.minor, label='', ticklabels=False)

        # phase
        pos = [left_offset/fig_width, phase_deriv_offset/fig_height,
               w/fig_width, phase_deriv_h/fig_height]
        ax_phase_deriv = fig.add_axes(pos)
        ax_phase_deriv.plot(p.tgrid, p.phase(unwrap=True, s=1000,
                            derivative=True))
        a = axes_info['time']
        if i_tgt < 4:
            set_axis(ax_phase_deriv, 'x', a.start, a.stop, step=a.step,
                     minor=a.minor, label='time (ns)', labelpad=1,
                     drop_ticklabels=[-1, ])
        else:
            set_axis(ax_phase_deriv, 'x', a.start, a.stop, step=a.step,
                     minor=a.minor, label='time (ns)', labelpad=1)
        a = axes_info['phase_deriv']['y']
        if i_tgt == 0:
            set_axis(ax_phase_deriv, 'y', a.start, a.stop, range=a.range,
                     step=a.step, minor=a.minor, label='')
        else:
            set_axis(ax_phase_deriv, 'y', a.start, a.stop, range=a.range,
                     step=a.step, minor=a.minor, label='', ticklabels=False)
        ax_phase_deriv.axhline(y=delta2, ls='--',
                               color=get_color('green'))
        ax_phase_deriv.axhline(y=delta1, ls='--',
                               color=get_color('orange'))
        ax_phase_deriv.axhline(y=(delta2+alpha2), ls='dotted',
                               color=get_color('green'))
        ax_phase_deriv.axhline(y=(delta1+alpha1), ls='dotted',
                               color=get_color('orange'))

        # pulse
        pos = [left_offset/fig_width, pulse_offset/fig_height,
               w/fig_width, pulse_h/fig_height]
        ax_pulse = fig.add_axes(pos)
        p.render_pulse(ax_pulse)
        avg_pulse = np.trapz(np.abs(p.amplitude), p.tgrid) / p.tgrid[-1]
        ax_pulse.axhline(y=avg_pulse, color='black', dashes=ls['dotted'])
        a = axes_info['time']
        set_axis(ax_pulse, 'x', a.start, a.stop, step=a.step, minor=a.minor,
                 label='', ticklabels=False, labelpad=1)
        a = axes_info['pulse']['y']
        if i_tgt == 0:
            set_axis(ax_pulse, 'y', a.start, a.stop, step=a.step,
                     minor=a.minor, label='')
        else:
            set_axis(ax_pulse, 'y', a.start, a.stop, step=a.step,
                     minor=a.minor, label='', ticklabels=False)

        # logical subspace population
        pos = [left_offset/fig_width,log_offset/fig_height,
               w/fig_width, log_h/fig_height]
        ax_log = fig.add_axes(pos)
        tgrid = np.genfromtxt(join(universal_rf[tgt], 'U_over_t.dat'),
                              usecols=(0, ), unpack=True)
        pop_loss = np.zeros(len(tgrid))
        for i, U in enumerate(
                get_prop_gate_of_t(join(universal_rf[tgt], 'U_over_t.dat'))):
            pop_loss[i] = U.pop_loss()
        avg_loss = np.trapz(pop_loss, tgrid) / tgrid[-1]
        ax_log.fill(tgrid, pop_loss, color=get_color('grey'))
        ax_log.axhline(y=avg_loss, color='black', dashes=ls['dotted'])
        a = axes_info['time']
        if i_tgt < 4:
            set_axis(ax_log, 'x', a.start, a.stop, step=a.step, minor=a.minor,
                     label='time (ns)', labelpad=1, drop_ticklabels=[-1, ])
        else:
            set_axis(ax_log, 'x', a.start, a.stop, step=a.step, minor=a.minor,
                     label='time (ns)', labelpad=1)
        a = axes_info['log']['y']
        if i_tgt == 0:
            set_axis(ax_log, 'y', a.start, a.stop, range=a.range, step=a.step,
                     minor=a.minor, label='')
        else:
            set_axis(ax_log, 'y', a.start, a.stop, range=a.range, step=a.step,
                     minor=a.minor, label='', ticklabels=False)

        # population dynamics
        def split_polar(U_over_t, U0_over_t):
            j_i_label = {
            '00_00': (0, 0), '00_01': (0, 1), '00_10': (0, 2), '00_11': (0, 3),
            '01_00': (1, 0), '01_01': (1, 1), '01_10': (1, 2), '01_11': (1, 3),
            '10_00': (2, 0), '10_01': (2, 1), '10_10': (2, 2), '10_11': (2, 3),
            '11_00': (3, 0), '11_01': (3, 1), '11_10': (3, 2), '11_11': (3, 3)}
            phase = defaultdict(list)
            r = defaultdict(list)
            for Ut in U_over_t:
                U = Ut.transpose()
                c_ff = np.diag(next(U0_over_t))  # field-free phase factor
                for label in j_i_label:
                    i, j = j_i_label[label]
                    assert (abs(c_ff[j]) - 1.0) < 1e-8
                    phase[label].append(
                            angle(U[i, j] * c_ff[j].conjugate()))
                    r[label].append(abs(U[i, j]))
            return phase, r

        phase, r = split_polar(
            get_prop_gate_of_t(join(universal_rf[tgt], 'U_over_t.dat')),
            get_prop_gate_of_t(join(field_free_rf[tgt], 'U_over_t.dat')))

        dyn_h_offset = 0.5*(w - 2*dyn_width)
        pos00 = [(left_offset+dyn_h_offset)/fig_width,
                 (dyn_offset+dyn_width)/fig_height,
                 dyn_width/fig_width, dyn_width/fig_height]
        pos01 = [(left_offset+dyn_h_offset+dyn_width)/fig_width,
                 (dyn_offset+dyn_width)/fig_height,
                 dyn_width/fig_width, dyn_width/fig_height]
        pos10 = [(left_offset+dyn_h_offset)/fig_width,
                 dyn_offset/fig_height,
                 dyn_width/fig_width, dyn_width/fig_height]
        pos11 = [(left_offset+dyn_h_offset+dyn_width)/fig_width,
                 dyn_offset/fig_height,
                 dyn_width/fig_width, dyn_width/fig_height]
        ax00 = fig.add_axes(pos00, projection='polar')
        ax01 = fig.add_axes(pos01, projection='polar')
        ax10 = fig.add_axes(pos10, projection='polar')
        ax11 = fig.add_axes(pos11, projection='polar')
        polar_axes.extend([ax00, ax01, ax10, ax11])
        if i_tgt == 0:
            fig.text((left_offset + dyn_h_offset-0.1)/fig_width,
                    (dyn_offset+dyn_width)/fig_height, rotation='vertical',
                    s=r'$\Im[\Psi(t)]$', verticalalignment='center',
                    horizontalalignment='right')
            fig.text((left_offset + dyn_h_offset-0.6)/fig_width,
                    (dyn_offset+0.25*dyn_width)/fig_height, rotation='vertical',
                    s=r'$\ket{00}$', verticalalignment='center',
                    horizontalalignment='right', color=get_color('blue'))
            fig.text((left_offset + dyn_h_offset-0.6)/fig_width,
                    (dyn_offset+0.75*dyn_width)/fig_height, rotation='vertical',
                    s=r'$\ket{01}$', verticalalignment='center',
                    horizontalalignment='right', color=get_color('orange'))
            fig.text((left_offset + dyn_h_offset-0.6)/fig_width,
                    (dyn_offset+1.25*dyn_width)/fig_height, rotation='vertical',
                    s=r'$\ket{10}$', verticalalignment='center',
                    horizontalalignment='right', color=get_color('red'))
            fig.text((left_offset + dyn_h_offset-0.6)/fig_width,
                    (dyn_offset+1.75*dyn_width)/fig_height, rotation='vertical',
                    s=r'$\ket{11}$', verticalalignment='center',
                    horizontalalignment='right', color=get_color('green'))
        fig.text((left_offset + dyn_h_offset + dyn_width)/fig_width,
                (dyn_offset-0.1)/fig_height,
                s=r'$\Re[\Psi(t)]$', verticalalignment='top',
                horizontalalignment='center')

        blue, orange, red, green = (get_color('blue'), get_color('orange'),
                                    get_color('red'), get_color('green'))
        ax00.plot(phase['00_00'], r['00_00'], color=blue,   lw=0.7)
        ax00.plot(phase['00_01'], r['00_01'], color=orange, lw=0.7)
        ax00.plot(phase['00_10'], r['00_10'], color=red,    lw=0.7)
        ax00.plot(phase['00_11'], r['00_10'], color=green,  lw=0.7)
        ax01.plot(phase['01_00'], r['01_00'], color=blue,   lw=0.7)
        ax01.plot(phase['01_01'], r['01_01'], color=orange, lw=0.7)
        ax01.plot(phase['01_10'], r['01_10'], color=red,    lw=0.7)
        ax01.plot(phase['01_11'], r['01_11'], color=green,  lw=0.7)
        ax10.plot(phase['10_00'], r['10_00'], color=blue,   lw=0.7)
        ax10.plot(phase['10_01'], r['10_01'], color=orange, lw=0.7)
        ax10.plot(phase['10_10'], r['10_10'], color=red,    lw=0.7)
        ax10.plot(phase['10_11'], r['10_11'], color=green,  lw=0.7)
        ax11.plot(phase['11_00'], r['11_00'], color=blue,   lw=0.7)
        ax11.plot(phase['11_01'], r['11_01'], color=orange, lw=0.7)
        ax11.plot(phase['11_10'], r['11_10'], color=red,    lw=0.7)
        ax11.plot(phase['11_11'], r['11_11'], color=green,  lw=0.7)
        ax00.scatter((phase['00_00'][0], ), (r['00_00'][0], ),
                     c=(get_color('blue'),), marker='s')
        ax00.scatter(
            (phase['00_00'][-1], phase['00_01'][-1], phase['00_10'][-1],
             phase['00_11'][-1]),
            (r['00_00'][-1], r['00_01'][-1], r['00_10'][-1], r['00_11'][-1]),
            c = [get_color(clr) for clr in ['blue', 'orange', 'red', 'green']],
            lw=0.5
        )
        ax01.scatter((phase['01_01'][0], ), (r['01_01'][0], ),
                     c=(get_color('orange'),), marker='s')
        ax01.scatter(
            (phase['01_00'][-1], phase['01_01'][-1], phase['01_10'][-1],
             phase['01_11'][-1]),
            (r['01_00'][-1], r['01_01'][-1], r['01_10'][-1], r['01_11'][-1]),
            c = [get_color(clr) for clr in ['blue', 'orange', 'red', 'green']],
            lw=0.5
        )
        ax10.scatter((phase['10_10'][0], ), (r['10_10'][0], ),
                     c=(get_color('red'),), marker='s')
        ax10.scatter(
            (phase['10_00'][-1], phase['10_01'][-1], phase['10_10'][-1],
             phase['10_11'][-1]),
            (r['10_00'][-1], r['10_01'][-1], r['10_10'][-1], r['10_11'][-1]),
            c = [get_color(clr) for clr in ['blue', 'orange', 'red', 'green']],
            lw=0.5
        )
        ax11.scatter((phase['11_11'][0], ), (r['11_11'][0], ),
                     c=(get_color('green'),), marker='s')
        ax11.scatter(
            (phase['11_00'][-1], phase['11_01'][-1], phase['11_10'][-1],
             phase['11_11'][-1]),
            (r['11_00'][-1], r['11_01'][-1], r['11_10'][-1], r['11_11'][-1]),
            c = [get_color(clr) for clr in ['blue', 'orange', 'red', 'green']],
            lw=0.5
        )

    for ax in polar_axes:
        ax.grid(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.plot(np.linspace(0, 2*np.pi, 50), np.ones(50)/np.sqrt(2.0), lw=0.5,
                dashes=ls['dashed'], color='black')
        ax.set_rmax(1.0)

    fig.text(y_label_offset/fig_width,
                (spec_offset+0.5*spec_h)/fig_height,
                r'$\vert F(\epsilon) \vert$ (arb. un.)',
                rotation='vertical', va='center', ha='left')
    fig.text(y_label_offset/fig_width,
                (phase_deriv_offset+0.5*phase_deriv_h)/fig_height,
                r'$\frac{d\phi}{dt}$ (MHz)',
                rotation='vertical', va='center', ha='left')
    fig.text(y_label_offset/fig_width,
                (pulse_offset+0.5*pulse_h)/fig_height,
                r'$\vert\epsilon\vert$ (MHz)',
                rotation='vertical', va='center', ha='left')
    fig.text(y_label_offset/fig_width,
                (log_offset+0.5*log_h)/fig_height,
                r'$P_\text{outside}$',
                rotation='vertical', va='center', ha='left')
    return fig



def latex_exp(f):
    """Convert float to scientific notation in LaTeX"""
    str = "%.1e" % f
    mantissa, exponent = str.split("e")
    return r'%.1f \times 10^{%d}' % (float(mantissa), int(exponent))


def generate_universal_pulse_plot_100ns(outfile):
    universal_root = './PLOT/'
    universal_rf = {
        'H_L': universal_root+'H_left',
        'H_R': universal_root+'H_right',
        'S_L': universal_root+'Ph_left',
        'S_R': universal_root+'Ph_right',
        'PE':  universal_root+'BGATE'
    }
    field_free_rf = {
        'H_L': universal_root+'fieldfree_2',
        'H_R': universal_root+'fieldfree_2',
        'S_L': universal_root+'fieldfree_2',
        'S_R': universal_root+'fieldfree_2',
        'PE':  universal_root+'fieldfree_c'
    }
    errors_diss = {
        'H_L': 7.2e-3, 'H_R': 7.3e-3, 'S_L': 7.4e-3, 'S_R': 7.7e-3,
        'PE':  7.4e-3,
    }

    errors_nodiss = {
        'H_L': 7.2e-3, 'H_R': 7.3e-3, 'S_L': 7.4e-3, 'S_R': 7.7e-3,
        'PE':  7.4e-3,
    }
    w_center = 5932.5
    w_1_dressed = 5982.5
    w_2_dressed = 5882.4
    wd = {
        'H_L': w_2_dressed, 'H_R': w_2_dressed,
        'S_L': w_2_dressed, 'S_R': w_2_dressed,
        'PE':  w_center,
    }
    axes_info = {
        'spec': {
            'x':  AttrDict(start=-200, stop=200, step=100, range=(-190, 190),
                           minor=10),
            'y':  AttrDict(start=0, stop=100, step=50, range=None, minor=2)
        },
        'time': AttrDict(start=0, stop=100, step=20, range=None, minor=2),
        'phase_deriv': {
            'y': AttrDict(start=-200, stop=200, step=100, range=(-190, 190),
                          minor=5),
        },
        'pulse': {
            'y': AttrDict(start=0, stop=200, step=100, range=None, minor=5),
        },
        'log': {
            'y': AttrDict(start=0, stop=0.1, step=0.05, range=(0, 0.12),
                          minor=5),
        },
    }
    fig = render_universal_pulse_plot(
        universal_rf, field_free_rf, errors_diss, errors_nodiss, wd, axes_info)
    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)
    fig.savefig(outfile, format=os.path.splitext(outfile)[1][1:])
    print("written %s" % outfile)


def main(argv=None):

    if argv is None:
        argv = sys.argv
    if not os.path.isdir(OUTFOLDER):
        QDYN.shutil.mkdir(OUTFOLDER)

    # Fig 5
    generate_universal_pulse_plot_100ns(outfile='fig5.pdf')

if __name__ == "__main__":
    sys.exit(main())
