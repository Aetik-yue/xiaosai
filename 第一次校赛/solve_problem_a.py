# /// script
# dependencies = [
#   "numpy",
#   "scipy",
#   "matplotlib",
#   "pulp"
# ]
# ///
import numpy as np
import matplotlib.pyplot as plt
import pulp

# Set plot style for academic paper
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] # Support Chinese labels
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 11})

# -------------------------------------------------------------
# 0. Load Data & Constants
# -------------------------------------------------------------
Tout_hourly = np.array([
    27.2, 26.8, 26.3, 25.9, 25.6, 25.8, 26.5, 27.9, 29.5, 31.0, 32.2, 33.1,
    33.8, 34.2, 34.5, 34.3, 33.9, 33.2, 32.4, 31.5, 30.6, 29.8, 29.0, 28.2
])
Tout_hourly_extended = np.append(Tout_hourly, 27.2)
t_hours = np.arange(25)
t_mins = np.arange(1440)
Tout_min = np.interp(t_mins / 60.0, t_hours, Tout_hourly_extended)

# System constants
EER = 3.5
Prated = 1.2 # kW
Tset_normal = 24.0
delta = 0.5 # deadband

# User group configurations
groups = {
    'A': {'R': 0.16, 'C': 550, 'Tup': 25.0, 'Tinit': 26.0, 'count': 300, 'allow_relax': False},
    'B_relax': {'R': 0.12, 'C': 600, 'Tup': 26.0, 'Tinit': 26.5, 'count': 400, 'allow_relax': True},
    'B_norelax': {'R': 0.12, 'C': 600, 'Tup': 26.0, 'Tinit': 26.5, 'count': 100, 'allow_relax': False},
    'C': {'R': 0.09, 'C': 700, 'Tup': 26.0, 'Tinit': 27.0, 'count': 200, 'allow_relax': True}
}

# -------------------------------------------------------------
# 1. Task 1: Single Household Modeling (B-Class)
# -------------------------------------------------------------
print("--- Running Task 1 ---")
# 1.1 Conventional Simulation (No Q_ambient for strict Task 1, as requested)
def sim_single_conventional(R, C, Tinit):
    a = np.exp(-60.0 / (1000.0 * R * C))
    b = R * 1000.0 * EER * (1.0 - a)
    
    Tin = np.zeros(1440)
    P = np.zeros(1440)
    Tin[0] = Tinit
    ac_state = 1
    
    for t in range(1440 - 1):
        if ac_state == 1:
            if Tin[t] <= Tset_normal - delta:
                ac_state = 0
        else:
            if Tin[t] >= Tset_normal + delta:
                ac_state = 1
        P[t] = Prated if ac_state == 1 else 0.0
        Tin[t+1] = a * Tin[t] + (1.0 - a) * Tout_min[t] - b * P[t]
        
    P[1439] = Prated if ac_state == 1 else 0.0
    return Tin, P

Tin_T1_conv, P_T1_conv = sim_single_conventional(groups['B_relax']['R'], groups['B_relax']['C'], groups['B_relax']['Tinit'])
P_T1_peak_conv_avg = np.mean(P_T1_conv[1080:1260])

# Plot Task 1 Conventional
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(t_mins / 60.0, Tin_T1_conv, 'b-', label='室内温度 $T_{in}$')
ax1.plot(t_mins / 60.0, Tout_min, 'r--', alpha=0.6, label='室外温度 $T_{out}$')
ax1.axhline(24.0, color='g', linestyle='--', label='设定温度 24°C')
ax1.fill_between(t_mins / 60.0, 23.5, 24.5, color='green', alpha=0.1, label='死区范围')
ax1.set_ylabel('温度 (°C)')
ax1.legend(loc='upper right')
ax1.set_title('B类用户（中保温）常规运行条件下的24小时变化曲线')
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(t_mins / 60.0, P_T1_conv, 'k-', label='空调功率 $P$')
ax2.set_ylabel('功率 (kW)')
ax2.set_xlabel('时间 (h)')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('task1_conv.png', dpi=300)
plt.close()
print("Saved task1_conv.png")

# 1.2 Pre-cooling optimization for B-class
# peak period is 18:00 - 21:00 (1080 to 1260)
# peak power limit: average power <= 0.7 * P_T1_peak_conv_avg = 0.014 kW
# peak temp limit: Tin <= 26.0 °C
# We allow pre-cooling: start time t_start in [12:00, 18:00] (720 to 1080)
# pre-cooling target temperature T_pre_min >= 18.0 °C
def sim_pre_cooling(t_start, T_pre_min):
    Tin = np.zeros(1440)
    P = np.zeros(1440)
    Tin[:t_start] = Tin_T1_conv[:t_start]
    P[:t_start] = P_T1_conv[:t_start]
    
    a = np.exp(-60.0 / (1000.0 * 0.12 * 600))
    b = 0.12 * 1000.0 * 3.5 * (1.0 - a)
    
    # Pre-cooling phase: set target to T_pre_min
    ac_state = 1
    for t in range(t_start, 1080):
        if ac_state == 1:
            if Tin[t] <= T_pre_min - delta:
                ac_state = 0
        else:
            if Tin[t] >= T_pre_min + delta:
                ac_state = 1
        P[t] = Prated if ac_state == 1 else 0.0
        Tin[t+1] = a * Tin[t] + (1.0 - a) * Tout_min[t] - b * P[t]
        
    # Peak phase: 18:00-21:00. Limit average power to 70% of baseline (which is 0.014 kW)
    P_limit = 0.7 * P_T1_peak_conv_avg
    for t in range(1080, 1260):
        P[t] = P_limit
        Tin[t+1] = a * Tin[t] + (1.0 - a) * Tout_min[t] - b * P[t]
        
    # After peak: restore conventional 24C
    ac_state = 1 if Tin[1260] > 24.0 else 0
    for t in range(1260, 1439):
        if ac_state == 1:
            if Tin[t] <= Tset_normal - delta:
                ac_state = 0
        else:
            if Tin[t] >= Tset_normal + delta:
                ac_state = 1
        P[t] = Prated if ac_state == 1 else 0.0
        Tin[t+1] = a * Tin[t] + (1.0 - a) * Tout_min[t] - b * P[t]
        
    P[1439] = Prated if ac_state == 1 else 0.0
    return Tin, P

# Run with t_start = 17:00 (minute 1020), T_pre_min = 20.0°C
Tin_pre, P_pre = sim_pre_cooling(1020, 20.0)
pre_energy = np.sum(P_pre[1020:1080]) / 60.0 # kWh
print("Pre-cooling phase energy (17:00-18:00): {:.4f} kWh".format(pre_energy))
print("Max temp during peak with pre-cooling: {:.4f} °C".format(np.max(Tin_pre[1080:1260])))

# Plot Task 1 Pre-cooling comparison
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_mins[900:1320] / 60.0, Tin_pre[900:1320], 'b-', label='预冷策略室内温度')
plt.plot(t_mins[900:1320] / 60.0, Tin_T1_conv[900:1320], 'g--', label='常规策略室内温度')
plt.axvline(18.0, color='r', linestyle=':', label='高峰开始 18:00')
ax = plt.gca()
ax.fill_between([18.0, 21.0], 15, 30, color='red', alpha=0.05, label='高峰时段 (18:00-21:00)')
plt.ylabel('温度 (°C)')
plt.ylim(19, 27)
plt.legend(loc='upper left')
plt.title('B类用户预冷调度与常规运行对比')
plt.grid(True, linestyle=':', alpha=0.6)

plt.subplot(2, 1, 2)
plt.plot(t_mins[900:1320] / 60.0, P_pre[900:1320], 'b-', label='预冷策略空调功率')
plt.plot(t_mins[900:1320] / 60.0, P_T1_conv[900:1320], 'g--', label='常规策略空调功率')
plt.axvline(18.0, color='r', linestyle=':')
plt.ylabel('功率 (kW)')
plt.xlabel('时间 (h)')
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('task1_precool.png', dpi=300)
plt.close()
print("Saved task1_precool.png")


# -------------------------------------------------------------
# 2. Task 2: Cluster Load Aggregation & Scheduling
# -------------------------------------------------------------
print("--- Running Task 2 ---")
# Conventional baseline for the 4 groups (strict ETP, Q_ambient = 0)
Tin_baseline = {}
P_baseline = {}
for g_name, g_info in groups.items():
    Tin_baseline[g_name], P_baseline[g_name] = sim_single_conventional(g_info['R'], g_info['C'], g_info['Tinit'])

# CBL (Customer Baseline Load) total is 1000 kW (representing high load peak on hot days)
CBL_total = 1000.0 # kW

# We solve the optimization problem for peak reduction target: 500 kW and 800 kW.
# Pre-cooling start time: 15:00 (minute 900) to 18:00 (1080).
# Decision variables: P_g(t) for t in 900 to 1259.
def solve_scheduling_dr(reduction_target):
    prob = pulp.LpProblem("AC_DR_Scheduling", pulp.LpMinimize)
    
    # T_vars spans 900 to 1260 (361 steps)
    t_T = list(range(900, 1261))
    # P_vars spans 900 to 1259 (360 steps)
    t_P = list(range(900, 1260))
    # Z_vars spans 1080 to 1260 (181 steps)
    t_Z = list(range(1080, 1261))
    
    P_vars = {}
    T_vars = {}
    Z_vars = {}
    
    for g_name in groups.keys():
        P_vars[g_name] = pulp.LpVariable.dicts(f"P_{g_name}", t_P, lowBound=0, upBound=Prated)
        T_vars[g_name] = pulp.LpVariable.dicts(f"T_{g_name}", t_T, lowBound=18.0)
        Z_vars[g_name] = pulp.LpVariable.dicts(f"Z_{g_name}", t_Z, lowBound=0.0)
        
    # Objective function: minimize pre-cooling energy + lambda * discomfort
    weight_discomfort = 15.0 # weighting parameter
    
    pre_energy_term = pulp.lpSum(groups[g_name]['count'] * P_vars[g_name][t] * (1.0/60.0)
                                 for g_name in groups.keys() for t in range(900, 1080))
    
    discomfort_term = pulp.lpSum(groups[g_name]['count'] * Z_vars[g_name][t] * (1.0/60.0)
                                 for g_name in groups.keys() for t in t_Z)
    
    prob += pre_energy_term + weight_discomfort * discomfort_term
    
    # Constraints
    for g_name, g_info in groups.items():
        R = g_info['R']
        C = g_info['C']
        a = np.exp(-60.0 / (1000.0 * R * C))
        b = R * 1000.0 * EER * (1.0 - a)
        
        # Initial temperature at 15:00
        prob += T_vars[g_name][900] == Tin_baseline[g_name][900]
        
        # Thermal dynamics
        for t in range(900, 1260):
            prob += T_vars[g_name][t+1] == a * T_vars[g_name][t] + (1.0 - a) * Tout_min[t] - b * P_vars[g_name][t]
            
        # Temperature bounds and discomfort variables
        for t in t_T:
            if t >= 1080: # Peak hours
                if g_info['allow_relax']:
                    T_max_allowed = 27.0
                else:
                    T_max_allowed = g_info['Tup']
            else: # Pre-cooling phase
                T_max_allowed = g_info['Tup']
                
            prob += T_vars[g_name][t] <= T_max_allowed
            
            # Discomfort z >= T - Tup
            if t >= 1080:
                prob += Z_vars[g_name][t] >= T_vars[g_name][t] - g_info['Tup']
                
    # Demand Response peak load constraint: Average power during peak hours <= CBL - reduction_target
    prob += pulp.lpSum(groups[g_name]['count'] * P_vars[g_name][t] * (1.0/180.0)
                       for g_name in groups.keys() for t in range(1080, 1260)) <= CBL_total - reduction_target
        
    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status_str = pulp.LpStatus[status]
    print(f"Target {reduction_target} kW - Solver Status:", status_str)
    
    # Extract results if feasible
    if status_str == 'Optimal':
        P_opt = {g: np.array([P_vars[g][t].varValue for t in t_P]) for g in groups.keys()}
        T_opt = {g: np.array([T_vars[g][t].varValue for t in t_T]) for g in groups.keys()}
        return P_opt, T_opt
    else:
        return None, None

P_opt_500, T_opt_500 = solve_scheduling_dr(500.0)
P_opt_800, T_opt_800 = solve_scheduling_dr(800.0)

# Generate and save scheduling plots for Task 2
def plot_scheduling(P_opt, T_opt, target_reduction, filename):
    if P_opt is None:
        print(f"Cannot plot for target {target_reduction} because solver was infeasible.")
        return
    t_T = np.arange(900, 1261)
    t_P = np.arange(900, 1260)
    
    # Calculate aggregate scheduled power
    P_dr_agg = np.zeros(len(t_P))
    P_base_agg = np.ones(len(t_P)) * 1000.0 # CBL baseline is constant 1000 kW during peak hours
    # Outside peak, baseline is the aggregate conventional power
    for t_idx, t in enumerate(t_P):
        if t < 1080:
            P_base_agg[t_idx] = sum(P_baseline[g][t] * groups[g]['count'] for g in groups.keys())
            
    # Also calculate scheduled aggregate power
    for g_name, g_info in groups.items():
        P_dr_agg += P_opt[g_name] * g_info['count']
        
    plt.figure(figsize=(10, 7))
    plt.subplot(2, 1, 1)
    for g_name, g_info in groups.items():
        plt.plot(t_T / 60.0, T_opt[g_name], label=f"{g_name} 室内温度 (上限 {g_info['Tup']}°C)")
    plt.axvline(18.0, color='r', linestyle=':', label='高峰开始 18:00')
    plt.ylabel('室内温度 (°C)')
    plt.legend(loc='lower left', fontsize=9)
    plt.title(f'空调集群在高峰时段平均削减 {target_reduction} kW 负荷的优化调度方案')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.subplot(2, 1, 2)
    plt.plot(t_P / 60.0, P_base_agg, 'g--', label='聚合常规基线负荷 (CBL)')
    plt.plot(t_P / 60.0, P_dr_agg, 'b-', label='聚合响应调度负荷')
    plt.fill_between(t_P / 60.0, P_dr_agg, P_base_agg, where=(t_P>=1080), color='blue', alpha=0.15, label='负荷削减量', step='mid')
    plt.axvline(18.0, color='r', linestyle=':')
    plt.ylabel('聚合总功率 (kW)')
    plt.xlabel('时间 (h)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

plot_scheduling(P_opt_500, T_opt_500, 500.0, 'task2_scheduling_500.png')
plot_scheduling(P_opt_800, T_opt_800, 800.0, 'task2_scheduling_800.png')
print("Saved task2_scheduling_500.png and task2_scheduling_800.png")

# Analyze user distribution and role differences
print("\n--- Scheduling Results Analysis ---")
for target, P_opt, T_opt in [(500, P_opt_500, T_opt_500), (800, P_opt_800, T_opt_800)]:
    if P_opt is None:
        continue
    print(f"\nReduction Target: {target} kW")
    total_pre_energy = 0.0
    total_discomfort = 0.0
    for g_name, g_info in groups.items():
        P_g = P_opt[g_name]
        T_g = T_opt[g_name]
        
        # Pre-cooling energy: sum_{t=0}^{179} P(t) * (1/60) * count
        g_pre_energy = np.sum(P_g[:180]) * (1.0/60.0) * g_info['count']
        total_pre_energy += g_pre_energy
        
        # Discomfort: sum_{t=180}^{360} (T(t) - Tup)^+ * (1/60) * count
        g_discomfort = np.sum(np.maximum(0, T_g[180:] - g_info['Tup'])) * (1.0/60.0) * g_info['count']
        total_discomfort += g_discomfort
        
        print(f"  Group {g_name} (n={g_info['count']}):")
        print(f"    Avg Pre-cooling Power: {np.mean(P_g[:180]):.4f} kW")
        print(f"    Avg Peak Temperature: {np.mean(T_g[180:]):.2f} °C (Max: {np.max(T_g[180:]):.2f} °C)")
        print(f"    Pre-cooling Energy: {g_pre_energy:.2f} kWh")
        print(f"    Discomfort: {g_discomfort:.2f} °C·h")
        
    print(f"  Aggregate Pre-cooling Energy: {total_pre_energy:.2f} kWh")
    print(f"  Aggregate Discomfort: {total_discomfort:.2f} °C·h")
    # Actual reduction
    P_dr_agg_peak = np.mean(sum(P_opt[g] * groups[g]['count'] for g in groups.keys())[180:])
    print(f"  Actual Average Peak Reduction: {CBL_total - P_dr_agg_peak:.2f} kW")


# -------------------------------------------------------------
# 3. Task 3: Robust Optimization under Uncertainty
# -------------------------------------------------------------
print("\n--- Running Task 3 ---")
np.random.seed(42)

def run_monte_carlo(P_opt_scheduled, T_opt_scheduled, num_trials=1000):
    actual_reductions = []
    
    # Precompute group dynamic constants (strict ETP, Q_ambient = 0)
    a_g = {}
    b_g = {}
    for g_name, g_info in groups.items():
        R = g_info['R']
        C = g_info['C']
        a_g[g_name] = np.exp(-60.0 / (1000.0 * R * C))
        b_g[g_name] = R * 1000.0 * EER * (1.0 - a_g[g_name])
        
    for trial in range(num_trials):
        # 1. Generate outdoor temperature noise epsilon(t) ~ N(0, 1.0)
        eps = np.random.normal(0, 1.0, 180)
        Tout_actual = Tout_min[1080:1260] + eps
        
        # Aggregate power under uncertainty
        P_actual_aggregate = 0.0
        
        for g_name, g_info in groups.items():
            count = g_info['count']
            exits = np.random.binomial(count, 0.05)
            actives = count - exits
            
            # Active users consume exactly the scheduled power
            P_active = P_opt_scheduled[g_name][180:360]
            
            # Exited users revert to conventional control under Tout_actual
            Tin_exit = np.zeros(180)
            P_exit = np.zeros(180)
            Tin_exit[0] = T_opt_scheduled[g_name][180]
            ac_state = 1 if Tin_exit[0] > 24.5 else 0
            
            for t in range(179):
                if ac_state == 1:
                    if Tin_exit[t] <= Tset_normal - delta:
                        ac_state = 0
                else:
                    if Tin_exit[t] >= Tset_normal + delta:
                        ac_state = 1
                P_exit[t] = Prated if ac_state == 1 else 0.0
                Tin_exit[t+1] = a_g[g_name] * Tin_exit[t] + (1.0 - a_g[g_name]) * Tout_actual[t] - b_g[g_name] * P_exit[t]
            P_exit[179] = Prated if ac_state == 1 else 0.0
            
            # Add this group's actual power to aggregate
            P_actual_aggregate += np.mean(actives * P_active + exits * P_exit)
            
        actual_reductions.append(CBL_total - P_actual_aggregate)
        
    return np.array(actual_reductions)

# Run Monte Carlo on deterministic 500 kW schedule
if P_opt_500 is not None:
    reductions_det_500 = run_monte_carlo(P_opt_500, T_opt_500, num_trials=1000)
    prob_success_det = np.mean(reductions_det_500 >= 500.0)
    print("Deterministic 500 kW Schedule - Probability of satisfying >= 500 kW reduction: {:.2%}".format(prob_success_det))
else:
    reductions_det_500 = None

# To find a robust strategy, we need to schedule for a higher target to absorb the uncertainty!
# Let's search for a robust target that gives >= 95% probability of success.
# Let's try robust target of 540 kW
print("Solving robust schedule with 540 kW target...")
P_opt_rob, T_opt_rob = solve_scheduling_dr(540.0)
if P_opt_rob is not None:
    reductions_rob = run_monte_carlo(P_opt_rob, T_opt_rob, num_trials=1000)
    prob_success_rob = np.mean(reductions_rob >= 500.0)
    print("Robust Schedule - Probability of satisfying >= 500 kW reduction: {:.2%}".format(prob_success_rob))
else:
    reductions_rob = None

# Save the probability distribution plot
if reductions_det_500 is not None and reductions_rob is not None:
    plt.figure(figsize=(10, 6))
    plt.hist(reductions_det_500, bins=30, alpha=0.6, color='red', edgecolor='black', label=f'确定性调度策略 (达标率: {prob_success_det:.2%})')
    plt.hist(reductions_rob, bins=30, alpha=0.6, color='blue', edgecolor='black', label=f'鲁棒调度策略 (达标率: {prob_success_rob:.2%})')
    plt.axvline(500.0, color='g', linestyle='--', linewidth=2, label='电网要求阈值 (500 kW)')
    plt.xlabel('高峰实际平均功率削减量 (kW)')
    plt.ylabel('频数 (模拟次数)')
    plt.title('确定性策略与鲁棒调度策略在多重不确定性下的实际平均削减量对比')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('task3_distribution.png', dpi=300)
    plt.close()
    print("Saved task3_distribution.png")

# Compare standard statistics between deterministic and robust
if reductions_det_500 is not None and reductions_rob is not None:
    print("\n--- Strategy Comparison ---")
    print("Deterministic Strategy:")
    print("  Success Probability (>=500 kW): {:.2%}".format(prob_success_det))
    print("  Mean Reduction: {:.2f} kW".format(np.mean(reductions_det_500)))
    print("  5th Percentile: {:.2f} kW".format(np.percentile(reductions_det_500, 5)))
    print("Robust Strategy:")
    print("  Success Probability (>=500 kW): {:.2%}".format(prob_success_rob))
    print("  Mean Reduction: {:.2f} kW".format(np.mean(reductions_rob)))
    print("  5th Percentile: {:.2f} kW".format(np.percentile(reductions_rob, 5)))
