import math
import torch
import torch.nn as nn
if __name__ == "__main__":
    import sys
    import os
    path = sys.path[0]
    path = path.split('/')
    path = '/'+os.path.join(*path[0:-1])
    sys.path.append(path)
    from baseline.baseAgent import baseAgent
else:
    from baseline.baseAgent import baseAgent


class Multi_Head_Attention(nn.Module):
    def __init__(
        self,
        M=8,
        device="cpu",
        use_tanh=False,
        C=None,
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
                self.d
            ).to(self.device)
        )

        self.w_query = nn.parameter.Parameter(
            torch.randn(
                self.d,
                self.d
            ).to(self.device)
        )
        self.w_values = nn.parameter.Parameter(
            torch.randn(
                self.d,
                self.d
            ).to(self.device)
        )
        # self.w_o = nn.parameter.Parameter(
        #     torch.randn(
        #         self.d_k,
        #         M,
        #         self.d
        #     ).to(self.device)
        # )
        self.dict_w_o = {}
        for i in range(self.M):
            self.dict_w_o[i] = nn.parameter.Parameter(
                torch.randn(
                    self.d_k,
                    self.d
                ).to(self.device)
            )

        self.categorical = torch.distributions.categorical.Categorical
        self.use_tanh = use_tanh
        self.C = C

    def forward_ENC(self, ref: torch.tensor, mask=None):
        # ref: B, N, D
        # mask: B, N (selecting means excluding) or B, N, N (adjacent matrix)
        BATCHSIZE, NODENUM = ref.shape[0], ref.shape[1]
        query = torch.matmul(ref, self.w_query).view(
            BATCHSIZE, NODENUM, self.M, self.d_k)
        key = torch.matmul(ref, self.w_key).view(
            BATCHSIZE, NODENUM, self.M, self.d_k)
        value = torch.matmul(ref, self.w_values).view(
            BATCHSIZE, NODENUM, self.M, self.d_k)
        # B, N, M, D
        query = query.permute(0, 2, 1, 3).contiguous()
        key = key.permute(0, 2, 1, 3).contiguous()
        value = value.permute(0, 2, 1, 3).contiguous()
        # B M N D

        query = torch.unsqueeze(query, dim=3)
        # B M N 1 D

        key = torch.unsqueeze(key, dim=2)
        # B M 1 N D

        compab = torch.sum(query * key, dim=-1) / (self.d_k) ** 0.5
        if mask is not None:
            compab = compab + (1 - mask.data) * (-1e10)
        compab = torch.softmax(compab, dim=-1)
        compab = torch.unsqueeze(compab, dim=-1)
        # B M N N 1

        value = torch.unsqueeze(value, dim=-3)
        # B M N 1 D

        h_m = torch.sum(compab * value, dim=-2)
        feature = torch.zeros(
            (BATCHSIZE, NODENUM, self.d)
        ).float().to(self.device)
        for i in range(self.M):
            feature = feature + torch.matmul(
                h_m[:, i, :, :], self.dict_w_o[i])

        return feature
        # B M N D

    def forward_DEC(self, ref, mask=None):
        pass


class Encoder:

    def __init__(
        self,
        device="cpu",
        M=8,
        dimension=128
    ):
        self.MHA = Multi_Head_Attention(
            device=device,
            M=M,
            dimension=dimension
        )
        self.BN_MHA = nn.BatchNorm1d(dimension)
        self.BN_FF = nn.BatchNorm1d(dimension)
        self.dimension = dimension
        cfg = {
            "module00": {
                "netCat": "MLP",
                "nLayer": 2,
                "iSize": dimension,
                "fSize": [512, 128],
                "act": ["relu", "linear"],
                "prior": 0,
                "BN": True,
                "bias": True,
                "input": [0],
                "output": True
            }
        }
        self.FF = baseAgent(cfg)

    def get_parameters(self) -> list:
        param = []

        param += list(self.BN_MHA.parameters())
        param += list(self.MHA.parameters())
        param += list(self.BN_FF.parameters())
        param += list(self.FF.getParameters())
        return param

    def forward(self, ref):
        MHA_output = self.MHA.forward_ENC(ref).view(-1, self.dimension)
        ref = ref.view(-1, self.dimension)
        output = self.BN_MHA.forward(ref + MHA_output)
        output = self.BN_FF.forward(output + self.FF.forward([output])[0])
        return output

    def mode(self, cond=True):
        self.BN_FF.train(cond)
        self.BN_MHA.train(cond)


class Decoder:

    def __init__(
            self,
            device="cpu",
            M=8,
            dimension=128*3,
            c=10):
        self.MHA = Multi_Head_Attention(
            device=device,
            M=M,
            dimension=dimension
        )
        self.MHA_Prob = Multi_Head_Attention(
            device=device,
            M=1,
            dimension=dimension,
            use_tanh=True,
            C=c
        )
        self.vl = nn.parameter.Parameter(
            torch.randn(
                1, dimension
            ).to(self.device)
        )
        self.vf = nn.parameter.Parameter(
            torch.randn(
                1, dimension
            ).to(self.device)
        )

    def forward(self, ref, pos=None):
        BATCHSIZE, NODENUM = ref.shape[0], ref.shape[1]
        # ref: B, N, D
        mask = torch.ones((BATCHSIZE, NODENUM))
        vls = torch.cat([self.vl for _ in range(BATCHSIZE*NODENUM)],
                        dim=0).view(BATCHSIZE, NODENUM, -1)
        vfs = torch.cat([self.vf for i in range(BATCHSIZE*NODENUM)],
                        dim=0).view(BATCHSIZE, NODENUM, -1)
        cat_ref = torch.cat([ref, vls, vfs], dim=-1)
        glim = self.MHA.forward_ENC(cat_ref, mask)
        prob, action = self.MHA_Prob.forward_ENC(glim, mask)


if __name__ == "__main__":
    BATCHSIZE = 32
    DIMENSION = 128
    NUMHEADER = 8
    NODENUM = 20

    enc = Encoder()
    ref = torch.randn((BATCHSIZE, NODENUM, DIMENSION))
    feature = enc.forward(ref)
    enc.get_parameters()
    enc.mode()
    enc.mode(False)

    # June 29, 2021
    # Question? How to handle a variable number of node.
    # To compute matrix multiplication, the shape of matrix should be static.
    # The simple answer is just to fix the shape of reference matrix and fill that matrix.
    # mask can be good solution to check whether for the partial of matrix to be valid.
    # The shape of mask can be variable, but always be identical in the same batch.
    # But the problem is that how to complete the matrix and the mask in parallel ways?
    # Also, there might be problems about choosing action.
    # TODO:
    # 1. variable shape of matrix.
    # 2. varialbe number of nodes.
