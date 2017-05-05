import QDYN
from oct import evaluate_pulse_rho, get_U, GATE

p_HL = QDYN.pulse.Pulse.read("./50NODISS/orig/pulse_HL.dat")
p_HR = QDYN.pulse.Pulse.read("./50NODISS/orig/pulse_HR.dat")
p_SL = QDYN.pulse.Pulse.read("./50NODISS/orig/pulse_SL.dat")
p_SR = QDYN.pulse.Pulse.read("./50NODISS/orig/pulse_SR.dat")
p_BG = QDYN.pulse.Pulse.read("./50NODISS/orig/pulse_BG.dat")
BGATE = QDYN.gate2q.Gate2Q.read("./50NODISS/orig/BGATE.dat")
p_HL_ff = p_HL.copy(); p_HL_ff.amplitude *= 0
p_HR_ff = p_HR.copy(); p_HR_ff.amplitude *= 0
p_SL_ff = p_SL.copy(); p_SL_ff.amplitude *= 0
p_SR_ff = p_SR.copy(); p_SR_ff.amplitude *= 0
p_BG_ff = p_BG.copy(); p_BG_ff.amplitude *= 0
U_HL = get_U(p_HL, 5932.5, dissipation=False, keep_runfolder="./PLOT50/H_left")
U_HR = get_U(p_HR, 5932.5, dissipation=False, keep_runfolder="./PLOT50/H_right")
U_SL = get_U(p_SL, 5932.5, dissipation=False, keep_runfolder="./PLOT50/Ph_left")
U_SR = get_U(p_SR, 5932.5, dissipation=False, keep_runfolder="./PLOT50/Ph_right")
U_BG = get_U(p_BG, 5932.5, dissipation=False, keep_runfolder="./PLOT50/BGATE")
U_ff = get_U(p_HL_ff, 5932.5, dissipation=False, keep_runfolder="./PLOT50/fieldfree_HL")
U_ff = get_U(p_HR_ff, 5932.5, dissipation=False, keep_runfolder="./PLOT50/fieldfree_HR")
U_ff = get_U(p_SL_ff, 5932.5, dissipation=False, keep_runfolder="./PLOT50/fieldfree_SL")
U_ff = get_U(p_SR_ff, 5932.5, dissipation=False, keep_runfolder="./PLOT50/fieldfree_SR")
U_ff = get_U(p_BG_ff, 5932.5, dissipation=False, keep_runfolder="./PLOT50/fieldfree_BG")

print("**** non-dissipative errors *****")
print("HL: %.2e" % (1-U_HL.F_avg(GATE['H_left'])))
print("HR: %.2e" % (1-U_HR.F_avg(GATE['H_right'])))
print("SL: %.2e" % (1-U_SL.F_avg(GATE['Ph_left'])))
print("SR: %.2e" % (1-U_SR.F_avg(GATE['Ph_right'])))
print("BG: %.2e" % (1-U_BG.F_avg(BGATE)))
#O = QDYN.weyl.closest_LI(U_BG, 0.5, 0.25, 0.0)
#print("BG: %.2e" % (1-U_BG.F_avg(O)))

print("")
answer = input("Do you want to obtain the fidelity in Liouville space? yes/[no]: ")
if answer.lower() == 'yes':
    print("")
    print("HL: %.2e" % (evaluate_pulse_rho(p_HL, GATE['H_left'], 5932.5, silent=True)))
    print("HR: %.2e" % (evaluate_pulse_rho(p_HR, GATE['H_right'], 5932.5, silent=True)))
    print("SL: %.2e" % (evaluate_pulse_rho(p_SL, GATE['Ph_left'], 5932.5, silent=True)))
    print("SR: %.2e" % (evaluate_pulse_rho(p_SR, GATE['Ph_right'], 5932.5, silent=True)))
    print("BG: %.2e" % (evaluate_pulse_rho(p_BG, BGATE, 5932.5, silent=True)))
    print("BG: %.2e" % (evaluate_pulse_rho(p_BG, O, 5932.5, silent=True)))
