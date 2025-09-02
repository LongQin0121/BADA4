import random
import copy
import numpy as np
import matplotlib.pyplot as plt


'''
智能优化算法大礼包【Python】遗传算法、蚁群优化算法、粒子群算法、禁忌搜索算法
https://download.csdn.net/download/qq_44186838/62610683
'''

#步骤1：贪心算法求初始路径距离
def Greedy(n,distance_graph):
    path=[0 for i in range(n+1)]  #记录已走城市下标
    distance=0     #记录贪心算法所得路径长
    i=1
    while i<=n:
        k=1
        if i==n:
            k=0
        Detemp=100
        while True:
            flag=0
            if k in path and k!=0:
                flag = 1
            if (flag==0) and (distance_graph[k][path[i-1]] < Detemp):
                j = k
                Detemp=distance_graph[k][path[i - 1]]
            k+=1
            if k>=n:
                break
        path.append(j)
        i+=1
        distance+=Detemp
    return(distance)


#步骤2：蚂蚁寻路径
class Ant(object):          #类的构造函数
    def __init__(self,ID):
        self.ID = ID                 # ID
        self.__clean_data()          # 随机初始化出生点
        
    # 步骤2.1 初始数据
    def __clean_data(self):
        self.path = []  # 当前蚂蚁的路径
        self.total_distance = 0.0  # 当前路径的总距离
        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的城市
        self.open_table_city = [True for i in range(n)]  # 探索城市的状态
        city_index = random.randint(0, n - 1)  # 随机初始出生点
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1

    # 步骤2.2 选择下一个城市
    def __choice_next_city(self):
        next_city = -1
        select_citys_prob = [0.0 for i in range(n)]  # 存储去下个城市的概率
        total_prob = 0.0

        # 获取去下一个城市的概率
        for i in range(n):
            if self.open_table_city[i]:
                # 计算概率：与信息素浓度成正比，与距离成反比
                try :
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], alpha) * pow((1.0/distance_graph[self.current_city][i]), beta)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print ('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID = self.ID, current = self.current_city, target = i))

        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(n):
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break
        if (next_city == -1):
            next_city = random.randint(0,n - 1)
            while ((self.open_table_city[next_city]) == False):  # if==False,说明已经遍历过了
                next_city = random.randint(0,n - 1)
            # 返回下一个城市序号
        return next_city

        # 移动操作
    def __move(self, next_city):

        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1

    #步骤2.4：构造路径
    def search_path(self):

        # 初始化数据
        self.__clean_data()

        # 搜素路径，遍历完所有城市为止
        while self.move_count < n:
            # 移动到下一个城市
            next_city = self.__choice_next_city()
            self.__move(next_city)
            

        # 计算路径总长度
        temp_distance = 0.0
        for i in range(1, n):
            start, end = self.path[i], self.path[i - 1]
            temp_distance += distance_graph[start][end]
        end = self.path[0]
        self.path.append(end)
        temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance


#步骤3：信息素更新
def __update_pheromone_gragh(ants):
 
    # 获取每只蚂蚁在其路径上留下的信息素
    temp_pheromone = [[0.0 for col in range(n)] for raw in range(n)]
    for ant in ants:
        for i in range(1,n):
            start, end = ant.path[i-1], ant.path[i]
            # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
            temp_pheromone[start][end] += 1.0 / ant.total_distance
            temp_pheromone[end][start] = temp_pheromone[start][end]

 
     # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
    for i in range(n):
        for j in range(n):
            pheromone_graph[i][j] = pheromone_graph[i][j] * rho + temp_pheromone[i][j]

#步骤4：多次迭代找路径
def search_path(ants,best_ant,N):
    iter=1
    y=[]
    while iter<=N:
    # 遍历每一只蚂蚁
        i=1
        for ant in ants:
            # 搜索一条路径
            ant.search_path()
            # 与当前最优蚂蚁比较
            if ant.total_distance < best_ant.total_distance:
                # 更新最优解
                best_ant =copy.deepcopy(ant)
            i+=1
    # 更新信息素
        __update_pheromone_gragh(ants)
        iter += 1
        y.append(int(best_ant.total_distance))
    x = np.linspace(1, N, N)
    plt.plot(x, y, 'r-')
    plt.xlabel('迭代次数')
    plt.ylabel('最优路径')
    plt.rcParams['font.sans-serif'] = ['SimHei']#解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('每次迭代最优路径图')
    plt.show()
    print ("迭代",N,"次后蚂蚁最佳路径为：",best_ant.path,"总距离：",int(best_ant.total_distance))

n=6    #城市数
m=10   #种群规模
N=100    #迭代次数
'''
alpha:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大
      ，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
beta:beta值越大，蚁群越就容易选择局部较短路径，这时算法收敛速度会
     加快，但是随机性不高，容易得到局部的相对最优
 rho:信息挥发因子，p过小会使路径残留信息素过多导致无效路径继续被搜索，
      p过大可能导致有效路径也被放弃搜索
 '''
alpha=1
beta=2
rho=0.5
distance_graph = [[1000,3,93,13,33,9],[4,1000,77,42,21,16],[45,17,1000,36,16,28],[39,90,80,1000,56,7],[28,46,88,33,1000,25],[3,88,18,46,92,1000]]# 城市距离
r0=m/Greedy(n,distance_graph)   #初始化信息素
pheromone_graph=[[r0 for col in range(n)] for raw in range(n)]#初始化信息素
ants = [Ant(ID) for ID in range(m)]  # 初始蚁群
best_ant = Ant(-1)                          # 初始最优解
best_ant.total_distance = 100         # 初始最大距离
search_path(ants,best_ant,N)