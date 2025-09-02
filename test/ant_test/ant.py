import numpy as np
import random

# 城市数量
n_cities = 5

# 距离矩阵（对称）
dist_matrix = np.array([
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 3],
    [9, 6, 0, 8, 5],
    [10, 4, 8, 0, 6],
    [7, 3, 5, 6, 0]
])

# 参数设置
n_ants = 10
n_iterations = 100
alpha = 1      # 信息素重要程度
beta = 2       # 启发因子重要程度
rho = 0.5      # 信息素挥发率
Q = 100        # 信息素常数

# 初始化信息素矩阵
pheromone = np.ones((n_cities, n_cities))

# 启发因子（1/距离）
eta = 1 / (dist_matrix + np.eye(n_cities))  # 防止除以0
np.fill_diagonal(eta, 0)

best_cost = float('inf')
best_path = []

for it in range(n_iterations):
    all_paths = []
    all_costs = []

    for _ in range(n_ants):
        path = [random.randint(0, n_cities - 1)]
        while len(path) < n_cities:
            current = path[-1]
            candidates = [i for i in range(n_cities) if i not in path]
            probs = []
            for j in candidates:
                tau = pheromone[current][j] ** alpha
                et = eta[current][j] ** beta
                probs.append(tau * et)
            probs = probs / np.sum(probs)
            next_city = random.choices(candidates, weights=probs)[0]
            path.append(next_city)
        path.append(path[0])  # 回到起点

        # 计算总路径成本
        cost = sum(dist_matrix[path[i]][path[i+1]] for i in range(n_cities))
        all_paths.append(path)
        all_costs.append(cost)

        if cost < best_cost:
            best_cost = cost
            best_path = path

    # 信息素更新
    pheromone *= (1 - rho)  # 蒸发
    for path, cost in zip(all_paths, all_costs):
        for i in range(n_cities):
            pheromone[path[i]][path[i+1]] += Q / cost

    if (it + 1) % 10 == 0:
        print(f"Iteration {it+1}: Best cost so far = {best_cost}")

print("\nBest path found:", best_path)
print("Best cost:", best_cost)
