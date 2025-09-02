Rosenbrock函数：优化理论中的经典测试函数
Rosenbrock函数是数学优化领域中最著名的非凸测试函数之一，因其独特的香蕉形谷底结构而广泛用于评估优化算法的性能。这个函数在机器学习、数值优化和算法开发中扮演着重要的基准测试角色，为研究人员提供了一个具有挑战性的优化问题。

https://www.bilibili.com/video/BV1i2421M7f6/?spm_id_from=333.337.search-card.all.click

代码工作原理解释
SGD优化过程：
初始化：设置初始点(1.3, 6.7)和学习率0.001

梯度计算：通过z.backward()自动计算∂z/∂x和∂z/∂y

参数更新：使用梯度下降公式更新参数

迭代：重复30000次直到收敛

关键步骤：
optimizer.zero_grad()：清空上一次的梯度

z.backward()：反向传播计算梯度

optimizer.step()：根据梯度更新参数

x.item()：将tensor转换为Python数值

这个实现展示了如何使用PyTorch的自动微分功能来优化Rosenbrock函数，SGD算法会逐步将参数从初始点(1.3, 6.7)优化到全局最优解(1.0, 1.0)附近。


wherever you are using SGD, use ADAM instead. 


性能特点对比
特性	SGD	Adam	L-BFGS
收敛速度	较慢	快速	非常快速
内存需求	低	中等	高
泛化能力	优秀	一般	良好
参数敏感性	高	低	中等
批处理支持	支持mini-batch	支持mini-batch	需要全批次