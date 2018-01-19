import numpy as np
import matplotlib.pyplot as plt


class Fisher:
    def __init__(self, group1Size, group2Size):
        self.group1 = []
        self.group2 = []
        self.group1Size = group1Size
        self.group2Size = group2Size
        self.covMatrix = np.matrix([[1, 0.2], [0.2, 1]])
        self.mu1 = np.matrix([3, 3])
        self.mu2 = np.matrix([8, 8])

    """Нормально распределенная группа"""
    def generateGroup(self, groupSize):
        self.groupSize = groupSize
        gr1 = np.random.normal(0, 1, groupSize)
        gr2 = np.random.normal(0, 1, groupSize)
        gr = np.vstack((gr1, gr2))
        return gr

    """Разложение Холецкого ковариационной матрицы"""
    @property
    def cholCovariance(self):
        matrix = np.linalg.inv(self.covMatrix)
        chol = np.linalg.cholesky(matrix)
        return chol

    """Преобрзованная с учетом центроида мю нормально распределенная группа"""
    def transformGroup(self, group, mu):
        group = self.cholCovariance*group + mu
        return group

    """Вектор из группы для расчета дискриминантной функции"""
    def getVector(self, group, n):
        self.group = group
        self.n = n
        vec = [group[0].tolist()[0][n], group[1].tolist()[0][n]]
        return vec

    """Коэффициенты дискриминантной функции"""
    def discriminantFunctionCoefficients(self, matrix, mu1T, mu2T):
        B = matrix*(mu1T - mu2T)
        return B

    """Дискриминантная функция"""
    @staticmethod
    def discriminantFunction(coefficients, groupVector):
        d = groupVector[0]*coefficients[0] + groupVector[1]*coefficients[1]
        return d


    def evaluate(self):
        normalGroup1 = self.generateGroup(500)
        normalGroup2 = self.generateGroup(500)
        group1 = self.transformGroup(normalGroup1, mu=np.transpose(self.mu1))
        group2 = self.transformGroup(normalGroup2, mu=np.transpose(self.mu2))
        vector1 = self.getVector(group1, 5)
        vector2 = self.getVector(group2, 5)
        B = self.discriminantFunctionCoefficients(self.cholCovariance, mu1T=np.transpose(self.mu1), mu2T=np.transpose(self.mu2))
        d_1 = self.discriminantFunction(B, vector1)
        d_2 = self.discriminantFunction(B, vector2)
        equidistantBoundary = (d_1 + d_2)/2
        x11 = 1
        x12 = (equidistantBoundary - B[0].tolist()[0][0]*x11)/B[1].tolist()[0][0]
        x21 = 10
        x22 = (equidistantBoundary - B[0].tolist()[0][0]*x21)/B[1].tolist()[0][0]
        x, y = [x11, x12], [x21, x22]

        for i in range(500):
            plt.plot(group1[0, i], group1[1, i], 'go', ms = 2)
            plt.plot(group2[0, i], group2[1, i], 'r^', ms = 6)

        plt.plot(x, y, 'k-')
        plt.show()




f = Fisher(500, 500)

if __name__ == '__main__':
    f.evaluate()





