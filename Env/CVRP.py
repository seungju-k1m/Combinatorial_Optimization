"""
Generate an environment for the Capacity Vehicle Route Problems.

CVRP can be represented by directed/undirected graph with cost and demand.

To define cost from node i to node j, QGIS, a kind of geographicla information system, is used.

Assumption

    Fleet is homoegenous.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np

import random
import math


class CVRP:
    def __init__(
        self,
        num_nodes: List[int, int]=None,
        num_vehicles: int=1,
        capacity_vehicle: float=10.0
    ):
        self.num_nodes = num_nodes
        self.num_vehicles = num_vehicles
        self.capacity_vehicle = capacity_vehicle

    def generate_without_QGIS(self):
        x_pos = np.array([random.random() for i in range(self.num_nodes)])
        y_pos = np.array([random.random() for i in range(self.num_nodes)])
        position = np.stack([x_pos, y_pos], axis=1)
        demand = [random.random() for i in range(self.num_nodes)]
        capacity = self.capacity_vehicle
        info = {"position": position, "demand": demand, "capacity": capacity}

        return info
