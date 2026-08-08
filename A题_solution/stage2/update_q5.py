"""
更新Q5: FY4→M2 + 重新计算并集 + 更新result3.xlsx
"""
import numpy as np, json, openpyxl
from core import *
from optimizer import calc_time_precise, calc_multi_bomb_time

def log(msg): print(msg, flush=True)

# 加载之前的结果
with open(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\results.json") as f:
    R = json.load(f)

# FY4新参数(逆向运动学找到的)
fy4_params = (np.radians(-54.87), 134.40, 4.849, 9.614)  # θ, v, t_rel, dt_fuse

# Q5新分配
assign = {
    'FY1': ('M1', R['Q5_params']['FY1']['params']),
    'FY2': ('M1', R['Q5_params']['FY2']['params']),
    'FY3': ('M2', R['Q5_params']['FY3']['params']),
    'FY4': ('M2', list(fy4_params)),
    'FY5': ('M3', R['Q5_params']['FY5']['params']),
}

log("="*60)
log("Q5 更新: FY4→M2")
log("="*60)

# 计算各弹精确遮蔽时长和区间
missile_intervals = {'M1': [], 'M2': [], 'M3': []}
all_results = {}

for uid, (mid, params) in assign.items():
    th, v, tr, df = params
    T, iv, bp, tb = calc_time_precise(mid, uid, th, v, tr, df)
    missile_intervals[mid].extend(iv)
    all_results[uid] = (mid, th, v, tr, df, T, iv, bp, tb)
    log(f"  {uid}→{mid}: T={T:.4f}s, θ={np.degrees(th):.2f}°, v={v:.1f}")
    log(f"    区间: {[(f'{s:.3f}',f'{e:.3f}') for s,e in iv]}")

# 各导弹遮蔽并集
log("\n各导弹遮蔽并集:")
total_T = 0.0
for mid in ['M1', 'M2', 'M3']:
    ivs = sorted(missile_intervals[mid])
    if not ivs:
        log(f"  {mid}: 0s (无遮蔽)")
        continue
    merged = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    union_T = sum(e - s for s, e in merged)
    total_T += union_T
    log(f"  {mid}: {union_T:.4f}s (合并{len(merged)}段)")

log(f"\nQ5总遮蔽: {total_T:.4f}s (之前: 15.149s)")

# 更新result3.xlsx
log("\n更新result3.xlsx...")
wb = openpyxl.Workbook(); ws = wb.active; ws.title="Sheet1"
ws.append(['无人机编号','无人机运动方向','无人机运动速度 (m/s)','烟幕干扰弹编号',
    '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
    '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
    '有效干扰时长 (s)','干扰的导弹编号'])

for uid in ['FY1','FY2','FY3','FY4','FY5']:
    mid, th, v, tr, df, T, iv, bp, tb = all_results[uid]
    drop = uav_pos(uid, th, v, tr)
    ws.append([uid, round(np.degrees(th)%360,4), round(v,4), 1,
               round(drop[0],4),round(drop[1],4),round(drop[2],4),
               round(bp[0],4),round(bp[1],4),round(bp[2],4), round(T,4), mid])
    for j in range(2):
        ws.append([uid, None, None, j+2]+[None]*8)

ws.append([None]*12)
ws.append([None,'注：以x轴为正向，逆时针方向为正，取值0~360（度）。']+[None]*10)
wb.save(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\result3.xlsx")
log("  result3.xlsx 已更新")

# 保存更新后的结果
q5_new = {
    'Q5_updated': round(total_T, 4),
    'details': {uid: {
        'missile': all_results[uid][0],
        'theta_deg': round(np.degrees(all_results[uid][1])%360, 4),
        'v': round(all_results[uid][2], 4),
        't_rel': round(all_results[uid][3], 4),
        'dt_fuse': round(all_results[uid][4], 4),
        'T': round(all_results[uid][5], 4),
    } for uid in all_results}
}
with open(r"C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2\q5_updated.json", 'w') as f:
    json.dump(q5_new, f, indent=2, ensure_ascii=False)
log("  q5_updated.json 已保存")

log(f"\n最终汇总:")
log(f"Q1: 1.392s, Q2: 4.578s, Q3: 6.405s, Q4: 11.506s, Q5: {total_T:.4f}s")
