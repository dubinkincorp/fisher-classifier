import numpy as np
import matplotlib.pyplot as plt


class Fisher:
    def __init__(self, group1Size, group2Size):
        self.group1 = []
        self.group2 = []
        self.group1Size = group1Size
        self.group2Size = group2Size
        self.covMatrix = np.matrix([[1, 0.2], [0.2, 1]])

    def generateGroup(self, groupSize):
        self.groupSize = groupSize
        gr1 = np.random.normal(0, 1, groupSize)
        gr2 = np.random.normal(0, 1, groupSize)
        gr = np.vstack((gr1, gr2))
        grT = np.transpose(gr)
        return grT

    def generateMeanVector(self, mu1, mu2):
        self.mu1 = mu1
        self.mu2 = mu2
        mu = np.matrix([mu1, mu2])
        return mu

    def transposeMeanVector(self, mu):
        self.mu = mu
        muT = np.transpose(mu)
        return muT

    @property
    def cholCovariance(self):
        chol = np.linalg.cholesky(self.covMatrix)
        return chol

    def calculateInvertibleMatrix(self, matrix):
        self.matrix = matrix
        invertibleMatrix = np.linalg.inv(matrix)
        return invertibleMatrix

    def getVector(self, group, n):
        self.group = group
        self.n = n
        vec = np.matrix([group[n].tolist()[0][0], group[n].tolist()[0][1]])
        vecT = np.transpose(vec)
        return vecT

    def discriminantFunction(self, mu, muT, matrix, x):
        self.mu = mu
        self.muT = muT
        self.matrix = matrix
        self.x = x
        gamma = 0.5*mu*matrix*muT
        d = mu*matrix*x - gamma
        return d

    def evaluate(self):
        group1 = self.generateGroup(500)
        group2 = self.generateGroup(500)
        mu1 = self.generateMeanVector(1, 1)
        mu2 = self.generateMeanVector(8, 8)
        mu1T = self.transposeMeanVector(mu1)
        mu2T = self.transposeMeanVector(mu2)
        cholCov = self.calculateInvertibleMatrix(self.cholCovariance)
        data1 = group1*cholCov + mu1
        data2 = group2*cholCov + mu2
        a1 = self.getVector(data1, 0)
        a2 = self.getVector(data1, 100)
        dA1 = self.discriminantFunction(mu1, mu1T, self.covMatrix, a1)
        dB1 = self.discriminantFunction(mu2, mu2T, self.covMatrix, a2)
        A = (mu1[0].tolist()[0][0] + mu2[0].tolist()[0][0])/2
        x = [mu1[0].tolist()[0][0]+A, dA1]
        y = [mu2[0].tolist()[0][0]+A, dB1]
        i = 0
        while i < len(data1):
            plt.plot(data1[i].tolist()[0][0], data1[i].tolist()[0][1], 'go', ms = 3)
            plt.plot(data2[i].tolist()[0][0], data2[i].tolist()[0][1], 'r^', ms = 6)
            i += 1

        plt.plot(x, y, 'ro')
        plt.plot(x, y, 'k-')
        plt.show()


f = Fisher(500, 500)

if __name__ == '__main__':
    f.evaluate()





