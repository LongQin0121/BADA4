import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# 创建优化变量
x = ca.SX.sym('x')
y = ca.SX.sym('y')
variables = ca.vertcat(x, y)

# 目标函数
objective = (x - 2) ** 2 + (y - 1) ** 2

# 约束条件
constraints = []
lb_constraints = []
ub_constraints = []

# 约束1: x^2 + y^2 <= 4
constraints.append(x ** 2 + y ** 2)
lb_constraints.append(-ca.inf)  # 下界为负无穷
ub_constraints.append(4)        # 上界为4

# 约束2: x + y >= 1
constraints.append(x + y)
lb_constraints.append(1)        # 下界为1
ub_constraints.append(ca.inf)   # 上界为正无穷

# 变量的边界 (x >= 0, y >= 0)
lb_variables = [0, 0]
ub_variables = [ca.inf, ca.inf]

# 将约束转化为向量
constraints_vector = ca.vertcat(*constraints)

# 创建NLP问题
nlp = {'x': variables, 'f': objective, 'g': constraints_vector}

# 使用SNOPT求解器（通过IPOPT接口，因为直接SNOPT接口可能需要额外安装）
# 在实际应用中，若您有SNOPT许可证，可以直接使用SNOPT
solver = ca.nlpsol('solver', 'ipopt', nlp)

# 求解NLP问题
solution = solver(x0=[0, 0],           # 初始猜测值
                  lbx=lb_variables,    # 变量下界
                  ubx=ub_variables,    # 变量上界
                  lbg=lb_constraints,  # 约束下界
                  ubg=ub_constraints)  # 约束上界

# 提取最优解
optimal_x = solution['x'].full().flatten()
optimal_value = solution['f'].full()[0][0]

print("最优解：")
print(f"x = {optimal_x[0]:.4f}")
print(f"y = {optimal_x[1]:.4f}")
print(f"目标函数值 = {optimal_value:.4f}")

# 检查约束条件
c1 = optimal_x[0]**2 + optimal_x[1]**2
c2 = optimal_x[0] + optimal_x[1]
print("\n约束条件检查：")
print(f"x^2 + y^2 = {c1:.4f} <= 4")
print(f"x + y = {c2:.4f} >= 1")

# 可视化结果
plt.figure(figsize=(8, 6))

# 绘制约束区域
theta = np.linspace(0, 2*np.pi, 100)
circle_x = 2*np.cos(theta)
circle_y = 2*np.sin(theta)
plt.plot(circle_x, circle_y, 'b--', label='x^2 + y^2 = 4')

x_vals = np.linspace(0, 4, 100)
y_vals = 1 - x_vals
valid_idx = y_vals >= 0
plt.plot(x_vals[valid_idx], y_vals[valid_idx], 'g--', label='x + y = 1')

plt.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='y = 0')
plt.axvline(x=0, color='r', linestyle='--', alpha=0.3, label='x = 0')

# 绘制目标点和最优解
plt.plot(2, 1, 'ro', label='目标点 (2,1)')
plt.plot(optimal_x[0], optimal_x[1], 'go', label=f'最优解 ({optimal_x[0]:.2f},{optimal_x[1]:.2f})')

# 绘制等高线
x_grid = np.linspace(0, 3, 100)
y_grid = np.linspace(0, 3, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = (X - 2)**2 + (Y - 1)**2
plt.contour(X, Y, Z, 20, alpha=0.5)

plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('非线性规划问题示例')
plt.legend()
plt.axis('equal')
plt.xlim(-0.5, 3)
plt.ylim(-0.5, 3)
plt.tight_layout()
plt.show()

# 软约束示例
print("\n软约束示例")
# 添加惩罚函数来实现软约束
penalty_weight = 100
soft_constraint_violation = ca.fmax(0, x**2 + y**2 - 4)  # 超出约束的部分
soft_objective = objective + penalty_weight * soft_constraint_violation

# 创建只有非负约束和x+y>=1的NLP问题（移除了圆的硬约束）
soft_constraints = [x + y]
soft_lb_constraints = [1]
soft_ub_constraints = [ca.inf]

soft_nlp = {'x': variables, 'f': soft_objective, 'g': ca.vertcat(*soft_constraints)}
soft_solver = ca.nlpsol('soft_solver', 'ipopt', soft_nlp)

# 求解软约束NLP问题
soft_solution = soft_solver(x0=[0, 0],
                           lbx=lb_variables,
                           ubx=ub_variables,
                           lbg=soft_lb_constraints,
                           ubg=soft_ub_constraints)

# 提取带软约束的最优解
soft_optimal_x = soft_solution['x'].full().flatten()
soft_optimal_value = soft_solution['f'].full()[0][0]

print("软约束下的最优解：")
print(f"x = {soft_optimal_x[0]:.4f}")
print(f"y = {soft_optimal_x[1]:.4f}")
print(f"目标函数值（含惩罚）= {soft_optimal_value:.4f}")