#!/usr/bin/env python3
"""HFE-7200 Closed-Loop Cool-Down Simulator — v3.4.1
===================================================
* Instantaneous heat gain from ambient and heat removed by the LN₂ heat-exchanger coil plots (both in W).
* Temperature-versus-time plot with uncertainty band on the top.

Changes since v3.4.0
--------------------
• Replaces the placeholder `C_tube_s` in the 1-D conduction-only model with the
  real coil UA calculated from the same resistances the main solver uses at
  t = 0 min under nominal conditions (2 L min⁻¹ HFE, h_boil = 1 000 W m⁻² K⁻¹).
"""
from __future__ import annotations
import math, numpy as np, matplotlib.pyplot as plt

# ─── Global toggles & materials ────────────────────────────────────────────
USE_INSULATION: bool = True
INS_THICK_M, INS_K = 0.05, 0.03
K_STEEL, CP_STEEL, H_AIR = 16.0, 500.0, 8.0

# ─── Geometry: tank, loop, coil ────────────────────────────────────────────
PROC_LEN = 4.5
PROC_OD  = 3/8*0.0254
PROC_WALL= 0.035*0.0254
PROC_ID  = PROC_OD - 2*PROC_WALL
LOOP_VOL_M3 = PROC_LEN*math.pi*(PROC_ID/2)**2

TANK_H, TANK_ID = 0.5, 0.096
TANK_WALL, TANK_OD = 0.0055, TANK_ID + 2*0.0055
TANK_VOL_M3 = math.pi*(TANK_ID/2)**2*TANK_H

TOTAL_VOL_M3 = LOOP_VOL_M3 + TANK_VOL_M3

HX_LEN, HX_OD = 3.8, 1/4*0.0254
HX_WALL, HX_ID = 0.049*0.0254, HX_OD - 2*0.049*0.0254
HX_SEG, HX_SEG_LEN = 50, HX_LEN/50
AREA_IN  = math.pi*HX_ID*HX_SEG_LEN
AREA_OUT = math.pi*HX_OD*HX_SEG_LEN
R_WALL_SEG = math.log(HX_OD/HX_ID)/(2*math.pi*K_STEEL*HX_SEG_LEN)

# ─── Steel masses for thermal capacity ─────────────────────────────────────
RHO_STEEL = 7_850
M_STEEL = (
    math.pi*((TANK_OD/2)**2-(TANK_ID/2)**2)*TANK_H +
    math.pi*((PROC_OD/2)**2-(PROC_ID/2)**2)*PROC_LEN +
    math.pi*((HX_OD/2)**2-(HX_ID/2)**2)*HX_LEN
)*RHO_STEEL

# ─── Stagnant HFE film around coil ─────────────────────────────────────────
STAGNANT_THICK_M, K_HFE_LIQ = 0.015, 0.065
R_HFE_STAGN_SEG = math.log(
    (HX_OD/2+STAGNANT_THICK_M)/(HX_OD/2)
)/(2*math.pi*K_HFE_LIQ*HX_SEG_LEN)

# ─── Simulation constants ─────────────────────────────────────────────────
T0, T_TARGET, T_AMB, T_LN2 = 298.0, 170.0, 293.0, 77.0
SIM_H, H_FG = 6.0, 1.99e5

SCEN = [
    ("Optimistic", 1500., 15., 3.0, 3.0),
    ("Nominal",    1000., 10., 1.0, 2.0),
    ("Pessimistic", 500.,  5., 0.1, 0.8),
]

# ─── Fluid properties ─────────────────────────────────────────────────────
def lpm_to_m3s(q):        return q/60000
def rho_hfe(T):           return 1420.0 - 2.9*(T-298)
def cp_hfe(T):            return 1220.0 + 1.5*(T-298)
def mu_hfe(T):            return 6.0e-4*(298/T)**1.5
def k_hfe(_):             return 0.065

def htc_hfe_ext(T, q_lpm):
    area = math.pi*(TANK_ID/2)**2
    v = lpm_to_m3s(q_lpm)/area
    Re = rho_hfe(T)*v*HX_OD/mu_hfe(T)
    Pr = cp_hfe(T)*mu_hfe(T)/k_hfe(T)
    if Re < 1e-6:
        return 0.0
    Nu = 0.3 + (0.62*Re**0.5*Pr**(1/3)) / (1+(0.4/Pr)**(2/3))**0.25 * (1+(Re/282000)**0.625)**0.8
    return Nu*k_hfe(T)/HX_OD

def cyl_u(L, r_i, wall, h_i):
    r_s = r_i+wall
    A_i = 2*math.pi*r_i*L
    R_i = 1/(h_i*A_i)
    R_steel = math.log(r_s/r_i)/(2*math.pi*K_STEEL*L)
    if USE_INSULATION:
        r_o = r_s + INS_THICK_M
        R_ins = math.log(r_o/r_s)/(2*math.pi*INS_K*L)
        R_o = 1/(H_AIR*2*math.pi*r_o*L)
        R_tot = R_i+R_steel+R_ins+R_o
    else:
        R_o = 1/(H_AIR*2*math.pi*r_s*L)
        R_tot = R_i+R_steel+R_o
    return 1/R_tot

C_AMB_TANK = cyl_u(TANK_H, TANK_ID/2, TANK_WALL, 150.)
C_AMB_LOOP = cyl_u(PROC_LEN, PROC_ID/2, PROC_WALL, 150.)
C_LEAK = C_AMB_TANK + C_AMB_LOOP
print("Ambient-loss C (W K⁻¹):", {k:round(v,3) for k,v in
      {"tank":C_AMB_TANK,"loop":C_AMB_LOOP,"tot":C_LEAK}.items()})

# ─── Core solver ──────────────────────────────────────────────────────────
RESULTS = {}
for lbl,h_boil,h_gas,fl_ln2,fl_hfe in SCEN:
    dt = TOTAL_VOL_M3 / lpm_to_m3s(fl_hfe)
    m_ln2 = lpm_to_m3s(fl_ln2)*808

    t_s,temp,hit = 0.0,T0,math.nan
    ts,Ts,Gs,Rs = [0.0],[temp],[0.0],[0.0]

    while t_s < SIM_H*3600:
        h_hfe = htc_hfe_ext(temp, fl_hfe)
        r_hfe = R_HFE_STAGN_SEG + 1/(h_hfe*AREA_OUT)

        r_b = 1/(h_boil*AREA_IN) + R_WALL_SEG + r_hfe
        r_g = 1/(h_gas *AREA_IN) + R_WALL_SEG + r_hfe
        r_b_tot,r_g_tot = r_b/HX_SEG, r_g/HX_SEG

        q_boil_max = (temp-T_LN2)/r_b_tot*dt
        q_lat = m_ln2*H_FG*dt
        q_rem = ( q_boil_max if q_lat>=q_boil_max else
                  q_lat + (1-q_lat/q_boil_max)*(temp-T_LN2)/r_g_tot*dt )

        q_gain = C_LEAK*(T_AMB-temp)*dt

        m_hfe = TOTAL_VOL_M3*rho_hfe(temp)
        cp_bulk = (m_hfe*cp_hfe(temp)+M_STEEL*CP_STEEL)/(m_hfe+M_STEEL)
        temp += (q_gain-q_rem)/((m_hfe+M_STEEL)*cp_bulk)

        t_s += dt
        ts.append(t_s/60); Ts.append(temp)
        Gs.append(q_gain/dt); Rs.append(q_rem/dt)
        if math.isnan(hit) and temp<=T_TARGET: hit = t_s/60

    RESULTS[lbl] = {"t":np.array(ts),"T":np.array(Ts),
                    "G":np.array(Gs),"R":np.array(Rs),"hit":hit}
    if lbl=="Nominal":
        print(f"Nominal → 170 K "
              f"{'reached' if math.isfinite(hit) else 'NOT reached'} "
              f"at {hit if hit==hit else SIM_H*60:.1f} min "
              f"(insul={'on' if USE_INSULATION else 'off'})")

# ─── 1-D conduction-only comparison ────────────────────────────────────────
SHOW_SIMPLE = True
if SHOW_SIMPLE:
    # --- Real coil UA for simple model (nominal point) --------------------
    H_BOIL_NOM, HFE_FLOW_NOM = 1_000.0, 2.0      # W m⁻² K⁻¹, L min⁻¹
    h_hfe_nom = htc_hfe_ext(T0, HFE_FLOW_NOM)
    r_hfe_nom = R_HFE_STAGN_SEG + 1/(h_hfe_nom*AREA_OUT)
    r_b_nom   = 1/(H_BOIL_NOM*AREA_IN) + R_WALL_SEG + r_hfe_nom
    r_b_tot_nom = r_b_nom / HX_SEG
    C_tube_s = 1 / r_b_tot_nom                    # ← proper UA
    C_amb_s  = C_LEAK

    def simple_curve():
        dt_s = TOTAL_VOL_M3 / lpm_to_m3s(2.0)
        t,T = 0.0,T0
        ts_s,Ts_s=[0.0],[T]
        while t < SIM_H*3600:
            q_gain = C_amb_s*(T_AMB-T)*dt_s
            q_rem  = C_tube_s*(T-T_LN2)*dt_s
            T += (q_gain-q_rem)/(rho_hfe(T)*TOTAL_VOL_M3*cp_hfe(T))
            t += dt_s
            ts_s.append(t/60); Ts_s.append(T)
        return np.array(ts_s),np.array(Ts_s)

    t_simple,T_simple = simple_curve()

# ─── Plotting ──────────────────────────────────────────────────────────────
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,8),sharex=True,
                           gridspec_kw={"height_ratios":[2,1]})

# add this:
fig_title = (
    "HFE-7200 Closed-Loop Cool-Down Simulation "
    f"(Insulation: {'50 mm polyurethane foam, k=0.03 W/m·K' if USE_INSULATION else 'Off'})"
)
fig.suptitle(fig_title, fontsize=16)

# then adjust your layout so the title isn’t cut off:
plt.tight_layout(rect=[0, 0, 1, 0.95])

ax1.plot(RESULTS["Nominal"]["t"],RESULTS["Nominal"]["T"],
         label="Nominal solver",color="C0")
if SHOW_SIMPLE:
    ax1.plot(t_simple,T_simple,'--',color="tab:green",
             label="1-D conduction model")
nt=RESULTS["Nominal"]["t"]
ax1.fill_between(
    nt,
    np.interp(nt,RESULTS["Optimistic"]["t"],RESULTS["Optimistic"]["T"]),
    np.interp(nt,RESULTS["Pessimistic"]["t"],RESULTS["Pessimistic"]["T"]),
    color="C0",alpha=0.3,label="Uncertainty band")
ax1.axhline(T_TARGET,ls="--",color="k",label="170 K target")
ax1.set(ylabel="Bulk HFE T (K)",ylim=(120,300)); ax1.grid(ls="--",alpha=0.5)
ax1.legend()

ax2.plot(nt,RESULTS["Nominal"]["G"],label="Heat gained (W)",color="tab:red")
ax2.plot(nt,RESULTS["Nominal"]["R"],label="Heat removed (W)",color="tab:blue")
ax2.set(xlabel="Time (min)",ylabel="Power (W)"); ax2.grid(ls="--",alpha=0.5)
ax2.legend()

plt.tight_layout(); plt.show()