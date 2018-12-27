import numpy as np
import matplotlib.pyplot as plt

class EvolutionComputing(object):
    def __init__(self, population_scale=100, parent_num=10, alpha=10e-14, algorithm = 'GT'):
        self.population_scale = population_scale
        self.parent_num = parent_num
        self.alpha = alpha
        self.algorithm = algorithm

    #初始化种群
    def init_population(self):
        self.X = np.zeros((self.population_scale, 5))
        for i in range(self.population_scale):
            self.X[i][0] = np.random.uniform(78, 102)
            self.X[i][1] = np.random.uniform(33, 45)
            for j in range(2, 5):
                self.X[i][j] = np.random.uniform(27, 45)

        self.current_population = self.X

    # 目标函数
    def object_func(self, x):
        return 5.3578547 * x[2] * x[2] + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141

    # 惩罚函数
    def __penalize(self, x):

        g1 = 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0]* x[3] - 0.0022053 * x[2] * x[4]
        h1 = 0
        if g1 > 92:
            h1 = g1 - 92
        elif g1 < 0:
            h1 = 0 - g1 

        g2 = 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] * x[2]
        h2 = 0
        if g2 > 110:
            h2 = g2 - 110
        elif g2 < 90:
            h2 = 90 - g2

        g3 = 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3]
        h3 = 0
        if g3 > 25:
            h3 = g3 -25
        elif g3 < 20:
            h3 = 20 - g3

        h4 = 0
        if x[0] > 102:
            h4 = x[0] - 102
        elif x[0] < 78:
            h4 = 78 - x[0]
        
        h5 = 0
        if x[1] > 45:
            h4 = x[1] - 45
        elif x[1] < 33:
            h4 = 33 - x[1]
        
        h6 = 0
        for i in range(2,5):
            if x[i] > 45:
                h6 = h6 + x[i] - 45
            elif x[i] < 27:
                h6 = h6 + 27 - x[i]

        return h1 + h2 + h3 + h4 + h5 + h6
    
    # 个体优劣的判断函数
    def better(self, x, y):
        """
        if x is better than y,return true
        x_p 惩罚值
        x_o 以x作为输入的目标函数值
        """
        x_p = self.__penalize(x)
        y_p = self.__penalize(y)
        if x_p < y_p:
            return True
        elif x_p > y_p:
            return False
        else:
            if self.object_func(x) <= self.object_func(y):
                return True
            else:
                return False

    # 在 X 群体中找到最差的的个体
    def get_worst(self, X):
        penalize_list = [self.__penalize(x) for x in X]
        sorted_index = np.argsort(penalize_list)
        worst_penalize = penalize_list[sorted_index[-1]]
        worst_p_index = np.where(penalize_list == worst_penalize)[0]
        if max(penalize_list) == 0:
            worst_index = np.argmax([self.object_func(x) for x in X])
            return worst_index
        if len(worst_p_index) > 1:
            d = {}
            for i, j in enumerate(worst_p_index):
                d[i] = j
            object_list = [self.object_func(x) for x in X[worst_p_index]]
            index_ = np.argsort(object_list)
            worst_parent_index = d[index_[-1]]
        else:
            worst_parent_index = sorted_index[-1]
        return worst_parent_index

    # 在 X 群体中找到最好的的个体
    def get_best(self, X):
        penalize_list = [self.__penalize(x) for x in X]
        sorted_index = np.argsort(penalize_list)

        min_p_index = []
        n = 0
        for m in penalize_list:
            if m == 0:
                min_p_index.append(n)
            n = n + 1

        if len(min_p_index) > 1:
            d = {}
            for i, j in enumerate(min_p_index):
                d[i] = j
            object_list = [self.object_func(x) for x in X[min_p_index]]
            index_ = np.argsort(object_list)
            best_index = d[index_[0]]
        else:
            best_index = sorted_index[0]
        return best_index

    # 采用MMEA算法中的多父体杂交的思想
    def __hybridize_MMEA(self):
        #产生 parent_num个父体
        indexs = [_ for _ in range(0, self.population_scale)]
        np.random.shuffle(indexs)
        indexs = indexs[0 : self.parent_num]
        parents = self.current_population[indexs]

        worst_parent_index = self.get_worst(parents)
        worst_parent = parents[worst_parent_index]
        #质心
        xp = (np.sum(parents, axis=0) - worst_parent) / (self.parent_num - 1)
        xr = 2 * xp - worst_parent
        sample = np.random.randint(1,4)
        
        if sample == 1:             #返回反射算子  
            return xr            
        elif sample == 2:           #返回压缩算子
            vec = xr - xp
            k = np.random.rand()
            xc = xp + vec * k
            return xc
        else:                       #返回扩张算子
            xz = 2 * xr - xp
            vec = xr - xz
            k = np.random.rand()
            xe = xz + vec * k
            return xe

    # 采用郭涛算法中的多父体杂交的思想
    def __hybridize_GT(self):
        #产生 parent_num个父体
        indexs = [_ for _ in range(0, self.population_scale)]
        np.random.shuffle(indexs)
        indexs = indexs[0 : self.parent_num]
        parents = self.current_population[indexs]

        # 在父体产生的 V 搜索空间内随机取一个个体
        V = []
        while True:
            a = np.random.uniform(-0.5, 1.5, self.parent_num - 1)
            sum = np.sum(a)
            b = 1 - sum
            if b >= -0.5 and b <= 1.5:
                a = np.hstack([a, b])
                x = np.array([0 for _ in range(5)])
                for i in range(0, self.parent_num):
                    x = x + parents[i] * a[i]
                V.append(x)
                if len(V) == 2:
                    break
        best_v = self.get_best(np.array(V))     
        return V[best_v]
    
    # 根用户的选择进行杂交计算
    def hybridize(self):
        if self.algorithm == 'GT':
            return self.__hybridize_GT()
        elif self.algorithm == 'MMEA':
            return self.__hybridize_MMEA()
        else:
            print('输入的算法名不对，可选择的有GT(郭涛算法) 和 MMEA算法')
            exit()
    
    # 种群进化，多父体杂交后产生的个体若比上一代种群的
    # 最差的个体要better，则替代之
    def evolute(self):
        self.init_population()
        best_index = 0
        best_index = self.get_best(self.current_population)
        worst_index = self.get_worst(self.current_population)
        gap = abs(self.object_func(self.current_population[best_index]) - self.object_func(self.current_population[worst_index]))
        best = self.current_population[best_index]
        worst = self.current_population[worst_index]
        x_point = []
        y_point = []

        iterations = 1
        while gap > self.alpha:
            x_new = self.hybridize()

            worst_index = self.get_worst(self.current_population)
            if self.better(x_new, self.current_population[worst_index]) == True:
                self.current_population[worst_index] = x_new

            gap = abs(self.object_func(self.current_population[best_index]) - self.object_func(self.current_population[worst_index]))
            print(gap)
            best_index = self.get_best(self.current_population)
            worst_index = self.get_worst(self.current_population)
            print(self.current_population[best_index])
            print("Generations " + str(iterations)+": " + str(self.object_func(self.current_population[best_index])))

            best = self.current_population[best_index]
            worst = self.current_population[worst_index]

            x_point.append(iterations)
            y_point.append(self.object_func(self.current_population[best_index]))
            iterations = iterations +1

        return best, x_point, y_point

                

if __name__ =='__main__':
    evolute_num = 20
    log_file = r'E:/学习/演化计算/GT.txt'
    optimal_values = []
    X = []
    X_Points = []
    Y_Points = []
    f = open(log_file, 'w')
    for i in range(evolute_num):
        e = EvolutionComputing(algorithm='GT', population_scale=100, parent_num=10)
        best, x_point, y_point = e.evolute()
        optimal_values.append(e.object_func(best))
        X.append(best)
        X_Points.append(x_point)
        Y_Points.append(y_point)

        f.write("===========第" + str(i) + "次实验结果===========")
        f.write("X: " + " ".join([str(x) for x in best]) + "\n")
        f.write("optimal_value: " + str(e.object_func(best)) + "\n")
        f.write(" ".join([str(_) for _ in x_point]) + "\n")
        f.write(" ".join([str(_) for _ in y_point]) + "\n\n")
    min_index = np.argmin(optimal_values)
    max_index = np.argmax(optimal_values)
    print("===========实验结果===========")
    print("最好结果 " + str(optimal_values[min_index]))
    print("平均 " + str(sum(optimal_values)/len(optimal_values)))
    print("最差结果 " + str(optimal_values[max_index]))
    print("最优值点为：")
    print(X[min_index])
    plt.plot(X_Points[min_index], Y_Points[min_index])
    plt.xlabel('iterations')
    plt.ylabel('optimal f')
    plt.show()

