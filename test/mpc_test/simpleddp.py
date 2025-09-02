import numpy as np
import matplotlib.pyplot as plt

class SimpleDDP:
    """
    最简单的DDP例子：一维小车位置控制
    状态：[位置, 速度]
    控制：[加速度]
    目标：从起点移动到目标位置并停下
    """
    
    def __init__(self):
        self.dt = 0.1        # 时间步长
        self.horizon = 20    # 预测步数（2秒）
        
        # 目标状态：位置=5，速度=0（到达目标点并停下）
        self.target = np.array([5.0, 0.0])
        
        # 权重（越大越重要）
        self.Q = np.array([[10.0, 0],    # 位置误差权重
                          [0, 1.0]])     # 速度误差权重
        self.R = np.array([[0.1]])       # 控制代价权重（用力越小越好）
        
        # 终端权重（最后时刻更重要）
        self.Qf = np.array([[100.0, 0],
                           [0, 10.0]])
    
    def dynamics(self, state, control):
        """
        小车动力学：简单的积分器模型
        状态 = [位置, 速度]
        控制 = [加速度]
        """
        position, velocity = state
        acceleration = control[0]
        
        # 下一时刻的状态
        new_position = position + velocity * self.dt
        new_velocity = velocity + acceleration * self.dt
        
        return np.array([new_position, new_velocity])
    
    def cost(self, state, control, is_final=False):
        """
        计算单步成本
        """
        error = state - self.target  # 与目标的误差
        
        if is_final:  # 最后一步
            return 0.5 * error.T @ self.Qf @ error
        else:  # 中间步骤
            state_cost = 0.5 * error.T @ self.Q @ error
            control_cost = 0.5 * control.T @ self.R @ control
            return state_cost + control_cost
    
    def solve_simple(self, start_state):
        """
        简化的DDP求解（只做几次迭代）
        """
        print("=== 简单DDP求解过程 ===")
        
        # 1. 初始化：生成初始轨迹
        states = np.zeros((self.horizon + 1, 2))
        controls = np.zeros((self.horizon, 1))
        
        states[0] = start_state
        print(f"起始状态: 位置={start_state[0]:.2f}, 速度={start_state[1]:.2f}")
        
        # 用简单的控制策略初始化
        for t in range(self.horizon):
            # 简单策略：朝目标方向用小力
            position_error = self.target[0] - states[t][0]
            controls[t] = np.array([np.clip(position_error * 0.5, -2, 2)])
            states[t + 1] = self.dynamics(states[t], controls[t])
        
        # 计算初始成本
        initial_cost = self.compute_total_cost(states, controls)
        print(f"初始总成本: {initial_cost:.2f}")
        
        # 2. DDP迭代改进
        for iteration in range(5):  # 只做5次迭代，方便观察
            print(f"\n--- 迭代 {iteration + 1} ---")
            
            # 反向传播：从终点开始计算最优策略
            feedback_gains = self.backward_pass_simple(states, controls)
            
            # 前向传播：应用新策略
            new_states, new_controls = self.forward_pass_simple(
                start_state, states, controls, feedback_gains)
            
            # 计算新成本
            new_cost = self.compute_total_cost(new_states, new_controls)
            improvement = initial_cost - new_cost
            
            print(f"新成本: {new_cost:.2f}, 改善: {improvement:.4f}")
            
            # 如果有改善，接受新轨迹
            if new_cost < initial_cost:
                states = new_states
                controls = new_controls
                initial_cost = new_cost
                print("✓ 接受新轨迹")
            else:
                print("✗ 拒绝新轨迹")
                break
        
        return states, controls
    
    def backward_pass_simple(self, states, controls):
        """
        简化的反向传播：计算反馈增益
        """
        gains = np.zeros((self.horizon, 1, 2))  # 每步的反馈增益
        
        # 从最后一步开始往前计算
        for t in range(self.horizon - 1, -1, -1):
            state = states[t]
            control = controls[t]
            
            # 简化：直接计算比例反馈增益
            # 这里用简单的启发式方法代替复杂的二次近似
            error = state - self.target
            
            # 位置误差大 -> 增加控制，速度大 -> 减少控制
            gains[t, 0, 0] = 2.0   # 位置反馈增益
            gains[t, 0, 1] = -1.0  # 速度反馈增益（阻尼）
        
        return gains
    
    def forward_pass_simple(self, start_state, old_states, old_controls, gains):
        """
        简化的前向传播：应用反馈控制
        """
        new_states = np.zeros_like(old_states)
        new_controls = np.zeros_like(old_controls)
        
        new_states[0] = start_state
        
        for t in range(self.horizon):
            # 计算状态误差
            state_error = new_states[t] - old_states[t]
            
            # 应用反馈控制：旧控制 + 反馈修正
            feedback = gains[t] @ state_error
            new_controls[t] = old_controls[t] + 0.5 * feedback  # 0.5是步长
            
            # 限制控制输入
            new_controls[t] = np.clip(new_controls[t], -3, 3)
            
            # 前向仿真
            new_states[t + 1] = self.dynamics(new_states[t], new_controls[t])
        
        return new_states, new_controls
    
    def compute_total_cost(self, states, controls):
        """
        计算轨迹的总成本
        """
        total_cost = 0
        
        # 中间步骤成本
        for t in range(self.horizon):
            total_cost += self.cost(states[t], controls[t])
        
        # 终端成本
        total_cost += self.cost(states[-1], np.array([0]), is_final=True)
        
        return total_cost
    
    def plot_results(self, states, controls):
        """
        绘制结果
        """
        time = np.arange(len(states)) * self.dt
        control_time = np.arange(len(controls)) * self.dt
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
        
        # 位置
        ax1.plot(time, states[:, 0], 'b-o', linewidth=2, markersize=4)
        ax1.axhline(y=self.target[0], color='r', linestyle='--', label='目标位置')
        ax1.set_ylabel('位置 (m)')
        ax1.set_title('小车位置轨迹')
        ax1.grid(True)
        ax1.legend()
        
        # 速度
        ax2.plot(time, states[:, 1], 'g-o', linewidth=2, markersize=4)
        ax2.axhline(y=self.target[1], color='r', linestyle='--', label='目标速度')
        ax2.set_ylabel('速度 (m/s)')
        ax2.set_title('小车速度轨迹')
        ax2.grid(True)
        ax2.legend()
        
        # 控制输入
        ax3.step(control_time, controls.flatten(), 'r-', linewidth=2, where='post')
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('加速度 (m/s²)')
        ax3.set_title('控制输入（加速度）')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    运行简单DDP示例
    """
    print("欢迎学习DDP！")
    print("问题：控制小车从起点移动到目标位置并停下")
    print("状态：[位置, 速度]")
    print("控制：[加速度]")
    print("目标：位置=5m, 速度=0m/s\n")
    
    # 创建DDP求解器
    ddp = SimpleDDP()
    
    # 设置起始状态
    start = np.array([0.0, 0.0])  # 位置=0, 速度=0
    
    # 求解
    final_states, final_controls = ddp.solve_simple(start)
    
    # 显示最终结果
    print(f"\n=== 最终结果 ===")
    print(f"最终位置: {final_states[-1, 0]:.2f} m (目标: {ddp.target[0]:.2f} m)")
    print(f"最终速度: {final_states[-1, 1]:.2f} m/s (目标: {ddp.target[1]:.2f} m/s)")
    print(f"位置误差: {abs(final_states[-1, 0] - ddp.target[0]):.3f} m")
    print(f"最大控制力: {np.max(np.abs(final_controls)):.2f} m/s²")
    
    # 绘制结果
    ddp.plot_results(final_states, final_controls)
    
    # 解释DDP的工作原理
    print("\n=== DDP工作原理解释 ===")
    print("1. 初始化：生成一个初始的控制轨迹")
    print("2. 反向传播：从终点开始，计算每一步的最优反馈策略")
    print("3. 前向传播：使用新的反馈策略生成改进的轨迹")
    print("4. 重复步骤2-3，直到收敛")
    print("\nDDP的优势：不仅给出控制序列，还给出反馈增益！")

if __name__ == "__main__":
    main()
