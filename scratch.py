from scipy.optimize import minimize
from QDYN.analytical_pulse import AnalyticalPulse
from math import atanh, tanh

from oct import *

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


def krotov_from_blackman(gate, wd, E0, T=100, iter_stop=100):
    from oct import krotov_from_pulse
    pulse = AnalyticalPulse(
            "blackman100ns", T=T, nt=2000, time_unit='ns',
            ampl_unit='MHz', parameters={'E0': E0}).to_num_pulse()
    return krotov_from_pulse(gate, wd, pulse, iter_stop=iter_stop)
