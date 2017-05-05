import QDYN
from oct import evaluate_pulse_rho, get_U, GATE

p_HL = QDYN.pulse.Pulse.read("/data/goerz/scratch/krotov_H_left/411fc0f1-31df-4141-8a96-ed4bab1fa401/pulse.oct.dat")
p_HR = QDYN.pulse.Pulse.read("/data/goerz/scratch/krotov_H_right/ca5bdeab-cc1b-461f-9b92-0d7ff7971490/pulse.oct.dat")
p_SL = QDYN.pulse.Pulse.read("/data/goerz/scratch/krotov_Ph_left/bb6747ff-e61b-4bc0-8d18-cd827703b09d/pulse.oct.dat")
p_SR = QDYN.pulse.Pulse.read("/data/goerz/scratch/krotov_Ph_right/d92ad594-bd6d-4962-aea1-591a894be257/pulse.oct.dat")
p_BG = QDYN.pulse.Pulse.read("/data/goerz/scratch/krotov_BGATE/5e468649-1b97-4d5e-aa3f-c32568482000/pulse.oct.dat")
p_ff_2 = QDYN.pulse.Pulse.read("/data/goerz/scratch/krotov_H_left/411fc0f1-31df-4141-8a96-ed4bab1fa401/pulse.oct.dat")
p_ff_c = QDYN.pulse.Pulse.read("/data/goerz/scratch/krotov_BGATE/5e468649-1b97-4d5e-aa3f-c32568482000/pulse.oct.dat")
BGATE = QDYN.gate2q.Gate2Q.read("/data/goerz/scratch/krotov_BGATE/5e468649-1b97-4d5e-aa3f-c32568482000/target_gate.dat")
p_ff_2.amplitude *= 0
p_ff_c.amplitude *= 0
U_HL = get_U(p_HL, 5882.4, dissipation=False, keep_runfolder="./PLOT100/H_left")
U_HR = get_U(p_HR, 5882.4, dissipation=False, keep_runfolder="./PLOT100/H_right")
U_SL = get_U(p_SL, 5882.4, dissipation=False, keep_runfolder="./PLOT100/Ph_left")
U_SR = get_U(p_SR, 5882.4, dissipation=False, keep_runfolder="./PLOT100/Ph_right")
U_ff = get_U(p_ff_2, 5882.4, dissipation=False, keep_runfolder="./PLOT100/fieldfree_2")
U_BG = get_U(p_BG, 5932.5, dissipation=False, keep_runfolder="./PLOT100/BGATE")
U_ff = get_U(p_ff_c, 5932.5, dissipation=False, keep_runfolder="./PLOT100/fieldfree_c")

print("**** non-dissipative errors *****")
print("HL: %.2e" % (1-U_HL.F_avg(GATE['H_left'])))
print("HR: %.2e" % (1-U_HR.F_avg(GATE['H_right'])))
print("SL: %.2e" % (1-U_SL.F_avg(GATE['Ph_left'])))
print("SR: %.2e" % (1-U_SR.F_avg(GATE['Ph_right'])))
print("BG: %.2e" % (1-U_BG.F_avg(BGATE)))
O = QDYN.weyl.closest_LI(U_BG, 0.5, 0.25, 0.0)
print("BG: %.2e" % (1-U_BG.F_avg(O)))

print("")
answer = input("Do you want to obtain the fidelity in Liouville space? yes/[no]: ")
if answer.lower() == 'yes':
    print("")
    print("HL: %.2e" % (evaluate_pulse_rho(p_HL, GATE['H_left'], 5882.4, silent=True)))
    print("HR: %.2e" % (evaluate_pulse_rho(p_HR, GATE['H_right'], 5882.4, silent=True)))
    print("SL: %.2e" % (evaluate_pulse_rho(p_SL, GATE['Ph_left'], 5882.4, silent=True)))
    print("SR: %.2e" % (evaluate_pulse_rho(p_SR, GATE['Ph_right'], 5882.4, silent=True)))
    print("BG: %.2e" % (evaluate_pulse_rho(p_BG, BGATE, 5932.5, silent=True)))
    print("BG: %.2e" % (evaluate_pulse_rho(p_BG, O, 5932.5, silent=True)))
