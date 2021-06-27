"""
Generate an environment for Euclidean 2D TSP(Travelling Sales Person).

The Environments consist of two elements, node and edge.

Node and Edge can be viewed respectively as city, and route.

To define the relationship between nodes and edges, you should define Distance Matrix.

In 2D Euclidean matric, distance matrix might be symmetric matrix.

If there are n nodes, there are n(n+1)/2 distance elements to define distance matrix.

"""
import matplotlib.pyplot as plt
import numpy as np

import random
import torch
import math


class Euclidean_2D_TSP_Env:

    def __init__(
        self,
        nodeNum: int,
        resolution=0.5,
        distanceMat=None,
        widHei=None
    ):
        self._nodeNum = nodeNum
        self._resolution = resolution
        if widHei is None:
            self._widHei = [self._nodeNum, self._nodeNum]
        else:
            self._widHei = widHei
        self._distanceMat = distanceMat

    @staticmethod
    def calculateEUD2D(x, y):
        return math.sqrt((x[0] - y[0])**2 + (x[1]-y[1])**2)

    @staticmethod
    def checkValidDistanceMatrix(mat, resolution=0.5):
        mask = mat > 0
        maskMat = mat[mask]
        if min(maskMat) < resolution:
            return False
        else:
            return True

    @staticmethod
    def append(posX, data):
        for d in data:
            posX.append(d)

    @staticmethod
    def plot(info, widHei=[10, 10]):
        print("Plotting")
        plt.plot(info["x_pos"], info["y_pos"], "bo")
        plt.grid(True)
        plt.xlim(0, widHei[0])
        plt.ylim(0, widHei[1])
        plt.axis('scaled')
        plt.title("TSP")
        plt.show()

    @staticmethod
    def plotLine(sequences, PI, widHei=[10, 10], t=1, title="TSP"):
        """
        input:
            sequence: (seq, batchSize, 2)
            PI: (batchSize, seq)
            widHei: [float, float]
        """
        with torch.no_grad():
            BATCHSIZE = sequences.shape[1]
            SEQ = PI.shape[1]
            for i in range(BATCHSIZE):
                plt.figure(1)
                sequence = sequences[:, i, :].numpy()
                policy = PI[i].numpy().astype(np.long)
                x_pos = sequence[:, 0]
                y_pos = sequence[:, 1]
                plt.plot(x_pos, y_pos, "bo")
                for j in range(SEQ-1):
                    #     plt.plot((x_pos[policy[j]], x_pos[policy[j+1]]),
                    #              (y_pos[policy[j]], y_pos[policy[j+1]]), 'r')
                    plt.arrow(x_pos[policy[j]], y_pos[policy[j]], x_pos[policy[j+1]] -
                              x_pos[policy[j]], y_pos[policy[j+1]] -
                              y_pos[policy[j]],
                              width=0.008, length_includes_head=True)
                # plt.plot((x_pos[policy[-1]], x_pos[policy[0]]),
                #          (y_pos[policy[-1]], y_pos[policy[0]]), 'r')
                plt.arrow(x_pos[policy[-1]], y_pos[policy[-1]], x_pos[policy[0]] -
                          x_pos[policy[-1]], y_pos[policy[0]] - y_pos[policy[-1]], width=0.008,
                          length_includes_head=True)
                
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.axis('scaled')
                plt.title(title)
                # plt.axes().set_aspect(1)
                plt.axis('equal')
                plt.show(block=False)
                plt.pause(t)
                plt.close()

    def generateRandom(self):
        # x_pos = [random.random() for i in range(self._nodeNum)]
        # y_pos = [random.random() for i in range(self._nodeNum)]
        x_pos, y_pos = [], []
        for i in range(self._nodeNum):
            if i % 2 == 0:
                x_pos.append(random.random())
                y_pos.append(1 - random.random())
            else:
                x_pos.append(1 - random.random())
                y_pos.append(random.random())
        random.shuffle(x_pos)
        random.shuffle(y_pos)
        xy_pos = np.array([x_pos, y_pos]).transpose(1, 0)
        return [], {"x_pos": x_pos, "y_pos": y_pos, "xy_pos": xy_pos}

    def generateRandomDistanceMatrix(self) -> np.ndarray:
        count = 0

        ratio = self._widHei[1] / self._widHei[0]
        boxNum = 10

        widBox = int(boxNum / (1 + ratio))
        heiBox = boxNum - widBox

        pt2Box = int(self._nodeNum / widBox / heiBox)
        if pt2Box < 2:
            pt2Box = self._nodeNum
            widBox = 1
            heiBox = 1
        remainder = self._nodeNum - widBox * heiBox * pt2Box

        widHei = [self._widHei[0] / widBox, self._widHei[1] / heiBox]

        x_pos, y_pos = [], []
        mat = np.zeros((self._nodeNum, self._nodeNum))
        for wid in range(widBox):
            for hei in range(heiBox):
                cond = False
                count += 1
                if count == boxNum:
                    pt2Box += remainder
                while cond is False:
                    x_pos_temp = [0.95 * widHei[0] * random.random() + wid * widHei[0]
                                  for _ in range(pt2Box)]
                    y_pos_temp = [0.95 * widHei[1] * random.random() + hei * widHei[1]
                                  for _ in range(pt2Box)]
                    for i in range(pt2Box):
                        for j in range(i, pt2Box):
                            value = self.calculateEUD2D([x_pos_temp[i], y_pos_temp[i]], [
                                                        x_pos_temp[j], y_pos_temp[j]])
                            mat[i + wid * pt2Box, j + hei * pt2Box] = value
                            mat[j + hei * pt2Box, i + wid * pt2Box] = value
                    cond = self.checkValidDistanceMatrix(
                        mat, resolution=self._resolution)
                    count += 1
                self.append(x_pos, x_pos_temp)
                self.append(y_pos, y_pos_temp)

        self._distanceMat = mat
        xy_pos = np.array([x_pos, y_pos]).transpose(1, 0)

        return mat, {"x_pos": x_pos, "y_pos": y_pos, "xy_pos": xy_pos}


if __name__ == "__main__":
    nodeNum = 30
    widHei = [20, 20]
    resolution = 1
    env = Euclidean_2D_TSP_Env(
        nodeNum=nodeNum, widHei=widHei, resolution=resolution)
    # mat, info = env.generateRandomDistanceMatrix()
    _, info = env.generateRandom()
    env.plot(info, widHei=widHei)
