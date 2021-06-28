import torch.nn as nn

import torch
import math
import os


class Multi_Head_Attention(nn.Module):
    def __init__(
        self,
        M=8,
        device="cpu",
        dimension=128
    ):
        super(Multi_Head_Attention, self).__init__()
        self.M = M
        self.d = dimension
        self.d_k = int(dimension/M)
        self.device = torch.device(device)
        self.w_key = nn.parameter.Parameter(
            torch.randn(
                self.d,
                self.d_k
            ).to(self.device)
        )

        self.w_query = nn.parameter.Parameter(
            torch.randn(
                self.d,
                self.d_k
            ).to(self.device)
        )
        self.w_values = nn.parameter.Parameter(
            torch.randn(
                self.d,
                M,
                self.d_k
            ).to(self.device)
        )
        self.dict_W_O = {}
        for i in range(self.M):
            setattr(
                self,
                "w_O_"+str(i),
                nn.parameter.Parameter(
                    torch.randn(
                        self.d_k,
                        self.d
                    ).to(self.device)
                ))
            self.dict_W_O[i] = getattr(self, "w_O_"+str(i))
        self.categorical = torch.distributions.categorical.Categorical

    def forward(self, ref: torch.tensor, mask: torch.tensor = None):
        # M개의 query와 key가 발생
        # M개의 attention을 구현
        # ref: batchSize, nodeNum,dimension
        # key: batchSize, nodeNum, d_M
        # query: batchSize, nodeNum, d_M
        # values: batchSize, nodeNum, M, d_M
        # mask: batchSize, nodeNum, nodeNum
        BATCHSIZE = ref.shape[0]
        NODENUM = ref.shape[1]
        key = torch.matmul(ref, self.w_key)
        query = torch.matmul(ref, self.w_query)
        values = torch.matmul(ref, self.w_values)

        # key와 query는 서로 내적 해야한다.
        # (1, d_M) * (nodeNum, d_M)
        key = torch.unsqueeze(key, dim=2)
        query = torch.unsqueeze(query, dim=1)
        # key: batchSize, nodeNum, <1, d_M>
        # query: batchSize, 1, <nodeNum, d_M>
        compability = torch.sum(key * query / (self.d_k) ** 0.5, -1)
        # compability: batchSize, nodeNum, nodeNum
        compability = compability * (1 - mask.data) ** (-1e10)
        prob = torch.softmax(compability, dim=-1)
        # prob: batchSize, nodeNum, nodeNum
        # ---------------calculate probability of each node--------------

        # prob * value
        # prob: batchSize, nodeNum, <nodeNum, 1>
        # value: batchSize, 1, <nodeNum, M * d_M>
        prob = torch.unsqueeze(prob, dim=-1)
        value = torch.unsqueeze(values, dim=1)
        feature = torch.sum(prob * value, 2)
        feature = feature.view(BATCHSIZE, NODENUM, self.M, self.d_k)
        output = torch.zeros((BATCHSIZE, NODENUM, self.d)
                             ).float().to(self.device)

        for i in range(self.M):
            feature_temp = feature[:, :, i, :]
            output += torch.matmul(feature_temp, self.dict_W_O[i])
        return output


class Encoder:

    def __init__(self):
        pass


class Decoder:

    def __init__(self):
        pass
