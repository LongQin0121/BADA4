from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

# 飞机数据
planes = ['A', 'B', 'C']
target_times = {'A': 10, 'B': 12, 'C': 14}
costs = {'A': 2, 'B': 3, 'C': 4}
min_sep = 3  # 最小间隔时间

# 创建问题实例
prob = LpProblem("Aircraft_Landing_Problem", LpMinimize)

# 定义变量：实际着陆时间和偏差
landing_times = LpVariable.dicts("LandingTime", planes, lowBound=0)
deviations = LpVariable.dicts("Deviation", planes, lowBound=0)

# 目标函数：最小化总偏差代价
prob += lpSum([costs[p] * deviations[p] for p in planes]), "Total_Deviation_Cost"

# 添加偏差约束
for p in planes:
    prob += landing_times[p] - target_times[p] <= deviations[p], f"Early_{p}"
    prob += target_times[p] - landing_times[p] <= deviations[p], f"Late_{p}"

# 添加最小间隔约束
for i in range(len(planes)):
    for j in range(i + 1, len(planes)):
        p1 = planes[i]
        p2 = planes[j]
        # 添加两个方向的间隔约束
        prob += landing_times[p1] + min_sep <= landing_times[p2] + 1000 * (1 - 0), f"Sep_{p1}_{p2}_1"
        prob += landing_times[p2] + min_sep <= landing_times[p1] + 1000 * (1 - 0), f"Sep_{p1}_{p2}_2"

# 求解问题
prob.solve()

# 输出结果
print(f"状态: {LpStatus[prob.status]}")
for p in planes:
    print(f"飞机 {p}: 实际着陆时间 = {landing_times[p].varValue:.2f} 分钟, 偏差 = {deviations[p].varValue:.2f} 分钟")
print(f"总偏差代价: {value(prob.objective):.2f}")
