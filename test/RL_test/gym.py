import gym
from gym import spaces
import numpy as np

class MultiAircraftEnv(gym.Env):
    """
    简单的多飞机航线规划环境
    状态：每架飞机的位置和速度（x, y, speed）
    动作：速度调整（加速、减速、保持）
    奖励：燃油消耗（速度越快越耗油），保持安全距离奖励
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, n_aircraft=3, max_speed=10, min_distance=2.0):
        super(MultiAircraftEnv, self).__init__()
        
        self.n_aircraft = n_aircraft
        self.max_speed = max_speed
        self.min_distance = min_distance
        
        # 状态空间：每架飞机(x, y, speed)
        # 假设2D空间限制为0-100单位
        low = np.array([0, 0, 0] * n_aircraft)
        high = np.array([100, 100, max_speed] * n_aircraft)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # 动作空间：每架飞机速度调整：-1（减速），0（保持），1（加速）
        self.action_space = spaces.MultiDiscrete([3] * n_aircraft)
        
        self.state = None
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self):
        # 随机初始化飞机位置和速度
        positions = np.random.uniform(low=10, high=90, size=(self.n_aircraft, 2))
        speeds = np.random.uniform(low=1, high=self.max_speed/2, size=(self.n_aircraft, 1))
        self.state = np.hstack((positions, speeds)).flatten()
        self.step_count = 0
        return self.state
    
    def step(self, action):
        """
        action: 每架飞机的速度调整 [-1, 0, 1]
        """
        self.step_count += 1
        
        # 解包状态
        state_reshaped = self.state.reshape(self.n_aircraft, 3)
        positions = state_reshaped[:, :2]
        speeds = state_reshaped[:, 2]
        
        # 更新速度
        speeds = speeds + (action - 1)  # -1 -> -1, 0 -> 0, 1 -> +1
        speeds = np.clip(speeds, 0, self.max_speed)
        
        # 简单假设所有飞机沿x方向前进
        positions[:, 0] += speeds
        
        # 保证位置不出界
        positions = np.clip(positions, 0, 100)
        
        # 更新状态
        self.state = np.hstack((positions, speeds.reshape(-1,1))).flatten()
        
        # 计算奖励
        reward = 0
        # 燃油消耗惩罚，速度越大耗油越多
        reward -= np.sum(speeds) * 0.1
        
        # 安全距离惩罚
        for i in range(self.n_aircraft):
            for j in range(i+1, self.n_aircraft):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.min_distance:
                    reward -= 10  # 严重惩罚靠得太近
        
        done = self.step_count >= self.max_steps
        
        return self.state, reward, done, {}
    
    def render(self, mode='human'):
        positions = self.state.reshape(self.n_aircraft, 3)[:, :2]
        print(f"Step: {self.step_count} Positions:\n{positions}")
        
    def close(self):
        pass


if __name__ == "__main__":
    env = MultiAircraftEnv(n_aircraft=3)
    obs = env.reset()
    done = False
    
    while not done:
        # 随机动作示范
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
        print(f"Reward: {reward}\n")
