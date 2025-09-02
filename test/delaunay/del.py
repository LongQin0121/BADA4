import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# 生成一组二维点
points = np.array([
    [0, 0],
    [1, 0],
    [0.5, 1],
    [0.5, 0.3],  # 尝试放一个点在三角形外接圆内部试试
    [1, 1]
])

# 进行 Delaunay 三角剖分
tri = Delaunay(points)

# 可视化结果
plt.figure(figsize=(8, 6))
plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='blue')
plt.plot(points[:, 0], points[:, 1], 'o', color='red')

# 显示点编号
for i, (x, y) in enumerate(points):
    plt.text(x + 0.02, y + 0.02, str(i), fontsize=12, color='black')

# 可视化每个三角形的外接圆
def plot_circumcircle(a, b, c):
    A = np.array(a)
    B = np.array(b)
    C = np.array(c)

    # 边向量
    AB = B - A
    AC = C - A

    # 计算外接圆圆心
    AB_perp = np.array([-AB[1], AB[0]])
    AC_perp = np.array([-AC[1], AC[0]])
    
    mid_AB = (A + B) / 2
    mid_AC = (A + C) / 2

    # 解线性方程求圆心
    A_mat = np.vstack((AB_perp, -AC_perp)).T
    b_vec = mid_AC - mid_AB
    try:
        t = np.linalg.solve(A_mat, b_vec)
        center = mid_AB + t[0] * AB_perp
        radius = np.linalg.norm(center - A)
        circle = plt.Circle(center, radius, color='green', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_patch(circle)
    except np.linalg.LinAlgError:
        pass  # 忽略退化情况

# 为每个三角形绘制外接圆
for simplex in tri.simplices:
    plot_circumcircle(points[simplex[0]], points[simplex[1]], points[simplex[2]])

plt.gca().set_aspect('equal')
plt.title("Delaunay Triangulation with Circumcircles")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
