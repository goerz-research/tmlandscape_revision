#!/usr/bin/env python
import os
import sys
import QDYN
import numpy as np
import matplotlib
matplotlib.use('Agg')
from mgplottools.mpl import new_figure, set_axis, get_color, ls
"""
Plot dressed levels
"""

STYLE = 'paper.mplstyle'

OUTFOLDER = './paper_images'
#OUTFOLDER = '/Users/goerz/Documents/Papers/TransmonLandscape'


def generate_dressed_level_plot(horiz_file, vert_file, outfile):
    fig_width     = 16.8
    fig_height    =  6.2
    left_margin   =  1.5
    right_margin  =  2.0
    top_margin    =  0.75
    bottom_margin =  0.75
    gap           =  0.2 # horizontal gap between panels

    w1_x     = 6000.0 # MHz
    w2_x     = 5900.0 # MHz
    wc_x     = 6200.0 # MHz
    alpha1 = -290.0 # MHz
    alpha2 = -310.0 # MHz
    g      =   70.0 # MHz

    alpha = 0.5 * abs(alpha1 + alpha2)

    blue = '#377eb8'
    yellow = '#ff7f00'
    red = '#e41a1c'
    green = '#4daf4a'
    purple = '#984ea3'
    brown = '#a65628'
    pink = '#f781bf'

    h = float(fig_height - top_margin - bottom_margin) / 2.0
    w = float(fig_width - left_margin - right_margin - gap) / 2.0

    fig = new_figure(fig_width, fig_height, style=STYLE)

    y_label = r'\begin{center}energy shift\\(MHz)\end{center}'

    #####################

    x, E_01, E_10, E_11, E_20, E_02, E_cav, w1, w2, wc = np.genfromtxt(
        horiz_file, unpack=True)

    #####################
    pos = [left_margin/fig_width, (bottom_margin + h)/fig_height,
           w/fig_width, h/fig_height]
    ax = fig.add_axes(pos)

    ax.axhline(y=0, lw=0.5, color='gray')
    ax.axvline(x=((w2_x-w1_x)/g), lw=0.5, color='gray')
    ax.axvline(x=0, lw=0.5, color='gray')

    ax.plot(x, E_01 - w2, label=r'$\Delta E_{01}$', color=green)
    ax.plot(x, E_10 - w1, label=r'$\Delta E_{10}$', color=purple,
            dashes=ls['solid'])
    ax.plot(x, E_cav - wc, label=r'$\Delta E_{\text{cav}}$', color=red,
            dashes=ls['dotted'])
    #ax.plot(x, E_20 - 2 * w1 - alpha1, label='$\Delta E_{20}$') # XXX
    #ax.plot(x, E_02 - 2 * w2 - alpha2, label='$\Delta E_{02}$') # XXX

    ax.axvline(x=((wc_x-w1_x)/g), ls='dashed', color='black')
    #ax.grid()

    ax.tick_params('y', which='both', right=True)
    set_axis(ax, 'y', -80, 80, range=(-100, 100), step=40, minor=4,
             label=y_label)
    set_axis(ax, 'x', -10, 10, range=(-9, 9), step=2, minor=2, label='',
             ticklabels=False)
    ax.axvspan(1, 9, color='black', alpha=0.1)

    #ax.annotate("horizontal cut", xy=(1, 1), xycoords="axes fraction",
                #xytext=(-5, -5), textcoords='offset points',
                #va="top", ha="right")

    ax2 = ax.twiny()
    set_axis(ax2, 'x', 5400, 6600, range=(w1_x-9*g, w1_x+9*g), step=200,
             minor=2, label=r'$\omega_c$ (MHz)')

    #####################
    pos = [left_margin/fig_width, bottom_margin/fig_height,
           w/fig_width, h/fig_height]
    ax = fig.add_axes(pos)

    ax.axhline(y=0, lw=0.5, color='gray')
    ax.axvline(x=((w2_x-w1_x)/g), lw=0.5, color='gray')
    ax.axvline(x=0, lw=0.5, color='gray')

    ax.plot(x, E_01 + E_10 - w1 - w2, label='$\Delta E_{01}+\Delta E_{10}$',
            color=blue, dashes=ls['dotted'])
    ax.plot(x, E_11 - w1 - w2, label='$\Delta E_{11}$', color=yellow)

    ax.axvline(x=((wc_x-w1_x)/g), ls='dashed', color='black')
    #ax.grid()

    ax.annotate(
        "", xy=(-1.8, -20), xycoords='data', xytext=(-1.8, 75),
        textcoords='data', arrowprops=dict(arrowstyle="<->",
        connectionstyle="arc3"))
    ax.annotate("$\zeta$", xy=(-1.5, 27.5), xycoords="data",
                va="center", ha="center",
                bbox=dict(boxstyle='square,pad=0', fc='white', ec='none'))
    ax.annotate("$\omega_1 \equiv$ 6000 MHz", xy=(8, 24),
                xycoords="axes points", va="center", ha="left")
    ax.annotate("$\omega_2 \equiv$ 5900 MHz", xy=(8, 15),
                xycoords="axes points", va="center", ha="left")
    #ax.annotate("$\omega_c =$ 6200 MHz", xy=(3.0, 70), xycoords="data",
                #va="center", ha="left")

    ax.tick_params('y', which='both', right=True)
    set_axis(ax, 'y', -80, 80, range=(-100, 100), step=40, minor=4,
             label=y_label)
    set_axis(ax, 'x', -10, 10, range=(-9, 9), step=2, minor=2,
             label=r'$\Delta_c/g$', tickpad=2)
    ax.axvspan(1, 9, color='black', alpha=0.1)

    #####################

    x, E_01, E_10, E_11, E_20, E_02, E_cav, w1, w2, wc = np.genfromtxt(
        vert_file, unpack=True)

    #####################
    pos = [(left_margin+gap+w)/fig_width, (bottom_margin + h)/fig_height,
           w/fig_width, h/fig_height]
    ax = fig.add_axes(pos)

    ax.axhline(y=0, lw=0.5, color='gray')
    ax.axvline(x=((wc_x-w1_x)/alpha), lw=0.5, color='gray')
    ax.axvline(x=0, lw=0.5, color='gray')

    line_01, = ax.plot(x, E_01 - w2, label=r'$\Delta E_{01}$', color=green)
    line_10, = ax.plot(x, E_10 - w1, label=r'$\Delta E_{10}$', color=purple,
                       dashes=ls['solid'])
    line_cav, = ax.plot(x, E_cav - wc, label=r'$\Delta E_{\text{cav}}$',
                        color=red, dashes=ls['dotted'])
    #line_20, = ax.plot(x, E_20 - 2 * w1 - alpha1, label='$\Delta E_{20}$') # XXX
    #line_02, = ax.plot(x, E_02 - 2 * w2 - alpha2, label='$\Delta E_{02}$') # XXX

    ax.axvline(x=((w2_x-w1_x)/alpha), ls='dashed', color='black')
    ax.axvline(x=((w2_x-w1_x)/alpha), ls='dashed', color='black')

    legend_top = ax.legend(handles=[line_cav], loc='upper right', ncol=1, frameon=1)
    legend_top.get_frame().set_color('white')
    legend_top.get_frame().set_alpha(0)
    ax.add_artist(legend_top)
    legend_bottom = ax.legend(handles=[line_01, line_10], loc=(0.715, -0.02),
                              ncol=1)
    ax.add_artist(legend_bottom)
    #legend3 = ax.legend(handles=[line_20, line_02], loc=(0.02, -0.02), ncol=1) # XXX

    set_axis(ax, 'y', -80, 80, range=(-100, 100), step=40, minor=4,
             label=y_label)
    set_axis(ax, 'x', -3, 3, range=(-2.5, 2.5), step=1, minor=2, label='',
             ticklabels=False)
    ax.tick_params('y', which='both', right=True, labelleft=False,
                   labelright=True)
    ax.yaxis.set_label_position("right")

    #ax.annotate("vertical cut", xy=(1, 1), xycoords="axes fraction",
                #xytext=(-5, -5), textcoords='offset points',
                #va="top", ha="right")

    ax2 = ax.twiny()
    set_axis(ax2, 'x', 5200, 6800, range=(w1_x-2.5*alpha, w1_x+2.5*alpha),
             step=200, minor=2, label=r'$\omega_2$ (MHz)')
    ax.axvspan(-0.8, 0.5, color='black', alpha=0.1)

    #####################
    pos = [(left_margin+gap+w)/fig_width, bottom_margin/fig_height,
           w/fig_width, h/fig_height]
    ax = fig.add_axes(pos)

    ax.axhline(y=0, lw=0.5, color='gray')
    ax.axvline(x=((wc_x-w1_x)/alpha), lw=0.5, color='gray')
    ax.axvline(x=0, lw=0.5, color='gray')

    line_sum, = ax.plot(x, E_01 + E_10 - w1 - w2,
                        label='$\Delta E_{01}+\Delta E_{10}$', color=blue,
                        dashes=ls['dotted'])
    line_11, = ax.plot(x, E_11 - w1 - w2, label='$\Delta E_{11}$',
                       color=yellow)

    ax.axvline(x=((w2_x-w1_x)/alpha), ls='dashed', color='black')

    legend_top = ax.legend(handles=[line_sum], ncol=1, frameon=1, borderpad=-0.25, loc=(0.605, 0.82))
    legend_top.get_frame().set_color('white')
    legend_top.get_frame().set_alpha(1)
    ax.add_artist(legend_top)
    legend_bottom = ax.legend(handles=[line_11], loc='lower right', ncol=1)

    ax.annotate("$\omega_1 \equiv$ 6000 MHz", xy=(8, 46),
                xycoords="axes points", va="center", ha="left")
                #bbox=dict(boxstyle='square,pad=0.02', fc='white', ec='none'))
    ax.annotate("$\omega_c \equiv$ 6200 MHz", xy=(8, 38),
                xycoords="axes points", va="center", ha="left")
                #bbox=dict(boxstyle='square,pad=0.02', fc='white', ec='none'))
    #ax.annotate("$\omega_2 =$ 5900 MHz", xy=(-0.3, 70), xycoords="data",
                #va="center", ha="left")

    set_axis(ax, 'y', -80, 80, range=(-100, 100), step=40, minor=4,
             label=y_label)
    set_axis(ax, 'x', -3, 3, range=(-2.5, 2.5), step=1, minor=2,
             label=r'$\Delta_2/\alpha$', tickpad=2)
    ax.tick_params('y', which='both', right=True, labelleft=False,
                   labelright=True)
    ax.yaxis.set_label_position("right")
    ax.axvspan(-0.8, 0.5, color='black', alpha=0.1)

    #####################
    fig.savefig(outfile, format=os.path.splitext(outfile)[1][1:])
    print("written %s" % outfile)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    if not os.path.isdir(OUTFOLDER):
        QDYN.shutil.mkdir(OUTFOLDER)
    outfile = os.path.splitext(os.path.basename(__file__))[0] + ".pdf"
    generate_dressed_level_plot(
        'dressed_levels_horiz_slice.dat', 'dressed_levels_vert_slice.dat',
        outfile)

if __name__ == "__main__":
    sys.exit(main())
