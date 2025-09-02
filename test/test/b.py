import numpy as np
import matplotlib.pyplot as plt

class AntColony:
    def __init__(self, distances, n_ants, n_iterations, decay=0.1, alpha=1, beta=2):
        """
        初始化蚁群算法
        :param distances: 城市间距离矩阵
        :param n_ants: 蚂蚁数量
        :param n_iterations: 迭代次数
        :param decay: 信息素衰减率
        :param alpha: 信息素重要程度因子
        :param beta: 启发式因子重要程度因子
        """
        self.distances = distances
        self.n_cities = len(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        # 初始化信息素矩阵
        self.pheromone = np.ones((self.n_cities, self.n_cities))
        self.best_path = None
        self.best_distance = float('inf')

    def run(self):
        for iteration in range(self.n_iterations):
            paths = self.construct_solutions()
            self.update_pheromone(paths)
            
            # 更新最佳路径
            iteration_best_path = paths[np.argmin([self.calculate_total_distance(p) for p in paths])]
            iteration_best_dist = self.calculate_total_distance(iteration_best_path)
            
            if iteration_best_dist < self.best_distance:
                self.best_distance = iteration_best_dist
                self.best_path = iteration_best_path
                
            print(f'迭代 {iteration + 1}, 最佳距离: {self.best_distance:.2f}')

    def construct_solutions(self):
        paths = []
        for ant in range(self.n_ants):
            path = self.construct_path()
            paths.append(path)
        return paths

    def construct_path(self):
        start_city = np.random.randint(self.n_cities)
        path = [start_city]
        available_cities = list(range(self.n_cities))
        available_cities.remove(start_city)
        
        while available_cities:
            current_city = path[-1]
            probabilities = self.calculate_probabilities(current_city, available_cities)
            next_city = np.random.choice(available_cities, p=probabilities)
            path.append(next_city)
            available_cities.remove(next_city)
            
        return path

    def calculate_probabilities(self, current_city, available_cities):
        pheromone = np.array([self.pheromone[current_city][j] for j in available_cities])
        distance = np.array([self.distances[current_city][j] for j in available_cities])
        
        # 计算启发式信息
        attractiveness = 1.0 / (distance + 1e-10)  # 添加小量避免除零
        
        # 计算概率
        probabilities = (pheromone ** self.alpha) * (attractiveness ** self.beta)
        probabilities = probabilities / probabilities.sum()
        
        return probabilities

    def update_pheromone(self, paths):
        # 信息素衰减
        self.pheromone *= (1.0 - self.decay)
        
        # 添加新的信息素
        for path in paths:
            distance = self.calculate_total_distance(path)
            for i in range(len(path)):
                current_city = path[i]
                next_city = path[(i + 1) % len(path)]
                self.pheromone[current_city][next_city] += 1.0 / distance
                self.pheromone[next_city][current_city] += 1.0 / distance

    def calculate_total_distance(self, path):
        total_distance = 0
        for i in range(len(path)):
            current_city = path[i]
            next_city = path[(i + 1) % len(path)]
            total_distance += self.distances[current_city][next_city]
        return total_distance

# 测试代码
def generate_cities(n_cities):
    """生成随机城市坐标"""
    cities = np.random.rand(n_cities, 2) * 100
    return cities

def calculate_distances(cities):
    """计算城市间距离矩阵"""
    n_cities = len(cities)
    distances = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            distances[i][j] = np.sqrt(np.sum((cities[i] - cities[j]) ** 2))
    return distances

def plot_path(cities, path):
    """绘制路径"""
    plt.figure(figsize=(10, 10))
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100)
    
    for i in range(len(cities)):
        plt.annotate(f'城市{i}', (cities[i][0], cities[i][1]))
    
    for i in range(len(path)):
        current_city = path[i]
        next_city = path[(i + 1) % len(path)]
        plt.plot([cities[current_city][0], cities[next_city][0]],
                [cities[current_city][1], cities[next_city][1]], 'b-')
    
    plt.title('蚁群算法解决TSP问题的路径')
    plt.show()

# 主程序
if __name__ == '__main__':
    # 生成10个随机城市
    n_cities = 10
    cities = generate_cities(n_cities)
    distances = calculate_distances(cities)
    
    # 创建蚁群算法实例
    aco = AntColony(
        distances=distances,
        n_ants=20,
        n_iterations=100,
        decay=0.1,
        alpha=1,
        beta=2
    )
    
    # 运行算法
    aco.run()
    
    # 绘制最优路径
    print(f'最优路径: {aco.best_path}')
    print(f'最短距离: {aco.best_distance:.2f}')
    plot_path(cities, aco.best_path)