import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cvxpy as cp

class ArrivalSequencingMPC:
    """
    民航进港航班间隔控制的线性MPC示例
    场景：多架飞机在同一进场航线上，需要维持安全间隔
    """
    
    def __init__(self, dt=10.0, horizon=20):
        self.dt = dt  # 时间步长（秒）
        self.horizon = horizon  # 预测时域
        
        # 系统参数
        self.min_separation = 6000  # 最小间隔距离（米）- 减少到6km
        self.nominal_speed = 80     # 标称进场速度（m/s）
        self.speed_bounds = [50, 120]  # 速度约束（m/s）- 放宽范围
        self.accel_bounds = [-3, 3]   # 加速度约束（m/s²）- 放宽范围
        
        # 权重矩阵
        self.Q_speed = 1.0      # 速度偏差权重
        self.Q_accel = 0.1      # 加速度权重
        self.Q_separation = 50.0  # 间隔维持权重 - 减少权重
        self.Q_slack = 1000.0   # 松弛变量权重
        
    def aircraft_dynamics(self, state, control):
        """
        飞机动力学模型（线性化）
        状态: [位置, 速度]
        控制: [加速度]
        """
        A = np.array([[1, self.dt],
                     [0, 1]])
        B = np.array([[0.5 * self.dt**2],
                     [self.dt]])
        
        return A @ state + B @ control
    
    def solve_mpc(self, aircraft_states, runway_position=0):
        """
        求解多机进港间隔控制MPC问题
        
        aircraft_states: list of [position, velocity] for each aircraft
        runway_position: 跑道位置（目标点）
        """
        n_aircraft = len(aircraft_states)
        n_states = 2  # [位置, 速度]
        n_controls = 1  # [加速度]
        
        # 定义优化变量
        states = []  # 每架飞机的状态轨迹
        controls = []  # 每架飞机的控制轨迹
        
        for i in range(n_aircraft):
            # 状态变量：[位置, 速度] over horizon
            x = cp.Variable((n_states, self.horizon + 1))
            # 控制变量：[加速度] over horizon  
            u = cp.Variable((n_controls, self.horizon))
            states.append(x)
            controls.append(u)
        
        # 构建约束和目标函数
        constraints = []
        objective = 0
        
        for i in range(n_aircraft):
            # 初始状态约束
            constraints.append(states[i][:, 0] == aircraft_states[i])
            
            # 动力学约束
            for k in range(self.horizon):
                A = np.array([[1, self.dt], [0, 1]])
                B = np.array([[0.5 * self.dt**2], [self.dt]])
                constraints.append(
                    states[i][:, k+1] == A @ states[i][:, k] + B @ controls[i][:, k]
                )
            
            # 控制约束
            for k in range(self.horizon):
                constraints.append(controls[i][0, k] >= self.accel_bounds[0])
                constraints.append(controls[i][0, k] <= self.accel_bounds[1])
            
            # 速度约束
            for k in range(self.horizon + 1):
                constraints.append(states[i][1, k] >= self.speed_bounds[0])
                constraints.append(states[i][1, k] <= self.speed_bounds[1])
            
            # 速度目标：维持标称进场速度
            for k in range(self.horizon + 1):
                objective += self.Q_speed * cp.square(states[i][1, k] - self.nominal_speed)
            
            # 控制平滑性
            for k in range(self.horizon):
                objective += self.Q_accel * cp.square(controls[i][0, k])
        
        # 间隔约束：使用松弛变量处理
        slack_vars = []
        for i in range(n_aircraft - 1):
            slack = cp.Variable(self.horizon + 1, nonneg=True)
            slack_vars.append(slack)
            
            for k in range(self.horizon + 1):
                # 后机与前机保持最小间隔（带松弛变量）
                separation = states[i][0, k] - states[i+1][0, k]
                constraints.append(separation + slack[k] >= self.min_separation)
                
                # 松弛变量惩罚
                objective += self.Q_slack * slack[k]
        
        # 求解优化问题
        prob = cp.Problem(cp.Minimize(objective), constraints)
        
        # 尝试使用已安装的求解器
        solvers_to_try = [cp.CLARABEL, cp.OSQP, cp.SCS]
        
        solved = False
        for solver in solvers_to_try:
            try:
                prob.solve(solver=solver, verbose=False)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    solved = True
                    print(f"使用求解器: {solver}")
                    break
            except:
                continue
        
        if not solved:
            # 如果指定求解器都失败，使用默认求解器
            prob.solve(verbose=False)
        
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            optimal_states = [s.value for s in states]
            optimal_controls = [u.value for u in controls]
            return optimal_states, optimal_controls, [s.value for s in slack_vars]
        else:
            print(f"优化求解失败: {prob.status}")
            return None, None, None
    
    def simulate_scenario(self):
        """模拟三架飞机进港场景"""
        # 初始状态：[位置(m), 速度(m/s)]
        # 重新设计初始状态，确保更合理的间隔
        aircraft_initial = [
            np.array([60000, 70]),  # 飞机1：距跑道60km，70m/s
            np.array([50000, 75]),  # 飞机2：距跑道50km，75m/s  
            np.array([40000, 80]),  # 飞机3：距跑道40km，80m/s
        ]
        
        # 检查初始间隔
        print("初始间隔检查：")
        for i in range(len(aircraft_initial) - 1):
            initial_sep = aircraft_initial[i][0] - aircraft_initial[i+1][0]
            print(f"  飞机{i+1} - 飞机{i+2}: {initial_sep/1000:.1f}km (最小要求: {self.min_separation/1000:.1f}km)")
        
        # 求解MPC
        opt_states, opt_controls, slack_values = self.solve_mpc(aircraft_initial)
        
        if opt_states is None:
            print("MPC求解失败，尝试调整参数...")
            return
        
        # 绘制结果
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        time_horizon = np.arange(0, (self.horizon + 1) * self.dt, self.dt)
        control_time = np.arange(0, self.horizon * self.dt, self.dt)
        
        colors = ['blue', 'red', 'green']
        aircraft_names = ['飞机1 (领先)', '飞机2 (中间)', '飞机3 (落后)']
        
        # 位置轨迹
        for i in range(len(opt_states)):
            ax1.plot(time_horizon, opt_states[i][0, :]/1000, 
                    color=colors[i], label=aircraft_names[i], linewidth=2)
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('距跑道距离 (km)')
        ax1.set_title('飞机位置轨迹')
        ax1.legend()
        ax1.grid(True)
        
        # 速度轨迹
        for i in range(len(opt_states)):
            ax2.plot(time_horizon, opt_states[i][1, :], 
                    color=colors[i], label=aircraft_names[i], linewidth=2)
        ax2.axhline(y=self.nominal_speed, color='black', linestyle='--', alpha=0.5, label='标称速度')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('速度 (m/s)')
        ax2.set_title('飞机速度轨迹')
        ax2.legend()
        ax2.grid(True)
        
        # 加速度控制输入
        for i in range(len(opt_controls)):
            ax3.step(control_time, opt_controls[i][0, :], 
                    color=colors[i], label=aircraft_names[i], linewidth=2, where='post')
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('加速度 (m/s²)')
        ax3.set_title('控制输入（加速度）')
        ax3.legend()
        ax3.grid(True)
        
        # 飞机间间隔
        for i in range(len(opt_states) - 1):
            separation = opt_states[i][0, :] - opt_states[i+1][0, :]
            ax4.plot(time_horizon, separation/1000, 
                    color=colors[i], label=f'{aircraft_names[i]} - {aircraft_names[i+1]}', 
                    linewidth=2)
        ax4.axhline(y=self.min_separation/1000, color='red', linestyle='--', 
                   label='最小安全间隔', alpha=0.7)
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('间隔距离 (km)')
        ax4.set_title('飞机间隔距离')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 打印关键信息
        print("\n=== 进港间隔控制MPC结果 ===")
        print(f"预测时域: {self.horizon} 步 ({self.horizon * self.dt} 秒)")
        print(f"最小安全间隔: {self.min_separation/1000:.1f} km")
        print(f"标称进场速度: {self.nominal_speed} m/s")
        
        print("\n初始状态:")
        for i, state in enumerate(aircraft_initial):
            print(f"  {aircraft_names[i]}: 位置 {state[0]/1000:.1f}km, 速度 {state[1]:.1f}m/s")
        
        print(f"\n{self.horizon * self.dt}秒后预测状态:")
        for i in range(len(opt_states)):
            final_pos = opt_states[i][0, -1]
            final_speed = opt_states[i][1, -1]
            print(f"  {aircraft_names[i]}: 位置 {final_pos/1000:.1f}km, 速度 {final_speed:.1f}m/s")
        
        # 检查间隔违规和松弛变量使用情况
        violations = 0
        total_slack = 0
        for i in range(len(opt_states) - 1):
            min_sep = np.min(opt_states[i][0, :] - opt_states[i+1][0, :])
            max_slack = np.max(slack_values[i]) if slack_values[i] is not None else 0
            total_slack += np.sum(slack_values[i]) if slack_values[i] is not None else 0
            
            if min_sep < self.min_separation:
                violations += 1
                print(f"⚠️  间隔违规: {aircraft_names[i]} - {aircraft_names[i+1]} 最小间隔 {min_sep/1000:.2f}km")
            
            if max_slack > 0.1:  # 显著的松弛变量使用
                print(f"📊 松弛变量使用: {aircraft_names[i]} - {aircraft_names[i+1]} 最大松弛 {max_slack/1000:.2f}km")
        
        print(f"总松弛变量使用: {total_slack/1000:.2f}km⋅时间步")
        
        if violations == 0:
            print("✅ 所有间隔约束均满足")

# 运行示例
if __name__ == "__main__":
    mpc_controller = ArrivalSequencingMPC(dt=10.0, horizon=20)
    mpc_controller.simulate_scenario()
