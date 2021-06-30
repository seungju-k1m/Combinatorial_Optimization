from baseline.utils import getOptim, writeTrainInfo, setup_logger
from baseline.baseAgent import baseAgent
from COP_RL.Config import NeuralCOCfg
from Env.TSP import Euclidean_2D_TSP_Env
from datetime import datetime
from itertools import count

import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np

# import _pickle
import logging
import random
import torch
import math
import time
import os


class Attention(nn.Module):
    def __init__(
        self,
        use_tanh=True,
        floatDim=16,
        q_dim=None,
        device="cpu"
    ):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.device = torch.device(device)
        v = torch.randn((floatDim, 1)).to(self.device)
        self.v = nn.parameter.Parameter(v)
        self.v.data.uniform_(-(1. / math.sqrt(floatDim)),
                             1. / math.sqrt(floatDim))
        self.w_ref = torch.randn(floatDim, floatDim).to(self.device)
        self.w_ref = nn.parameter.Parameter(self.w_ref)
        self.w_ref.data.uniform_(
            -(1. / math.sqrt(floatDim)),
            1. / math.sqrt(floatDim)
        )
        if q_dim is None:
            q_dim = floatDim
        # self.w_ref.requires_grad = False
        self.w_q = torch.randn(q_dim, floatDim).to(self.device)
        self.w_q = nn.parameter.Parameter(self.w_q)
        self.w_q.data.uniform_(
            -(1. / math.sqrt(floatDim)),
            1. / math.sqrt(floatDim)
        )
        # self.w_q.requires_grad = False
        # self.w_ref = nn.Linear(128, 128).to(self.device)
        # self.w_query = nn.Linear(128, 128).to(self.device)
        self.floatDim = floatDim
        self.categorical = torch.distributions.categorical.Categorical
        if self.use_tanh:
            self.C = 10
        else:
            self.C = 1

    def forward(self, query, ref, mask=None, temperature=1.0, sampleMode="Greedy"):
        """
            input:
                query: hidden state of decoder at time i, (1, batchSize, floatDim)
                ref: refercne vector [(1, batch, floatDim)]
                mask: masking for selected node. (seq, batchSize)
            output:
                prob: probability of selectiong node. (batchSize, seq)
                selectedNode: selected Node. (batchSize)
        """
        SEQ, BATCHSIZE = len(ref), query.shape[1]
        TEMPERATURE = temperature

        # U would be (SEQ, BATCHSIZE)
        U = torch.zeros((0, BATCHSIZE)).to(self.device)

        queryFeature = torch.matmul(query, self.w_q)[0]
        # for i in range(SEQ):
        #     feature = torch.matmul(
        #         torch.tanh((torch.matmul(ref[i], self.w_ref) + queryFeature)
        #                    ), self.v).view(1, -1)

        #     U = torch.cat([U, feature], dim=0)
        # SEQ, batch, dim
        U = torch.matmul(
            torch.tanh(
                (torch.matmul(ref, self.w_ref) +
                 torch.unsqueeze(queryFeature, dim=0))
            ), self.v
        ).view(SEQ, -1)

        if self.use_tanh:
            # U = (torch.exp(U/TEMPERATURE) - torch.exp(-U/TEMPERATURE)) / \
            #     (torch.exp(U/TEMPERATURE) + torch.exp(-U/TEMPERATURE))
            # U = (1 - torch.exp(-U/TEMPERATURE)) / \
            #     (1 + torch.exp(-U/TEMPERATURE))
            U = torch.tanh(U/TEMPERATURE)

        if mask is not None:
            U = U + (1 - mask.data) * (-1e10)
        prob = torch.softmax(U * self.C, dim=0)

        prob = prob.permute(1, 0).contiguous()
        # prob: (BATCHSIZE*ITER, SEQ)

        # -----------------------Sampling Mode ---------------------
        with torch.no_grad():
            probCpu = prob.cpu().detach()
            if sampleMode == "Greedy":
                # sample = self.categorical(prob).sample()
                # maximumValue, sample = prob.max(1)
                sample = torch.argmax(probCpu, dim=1)

            elif sampleMode == "Sampling":
                if True in torch.isnan(probCpu):
                    print("nan ")
                probCpu[torch.isnan(probCpu)] = 0

                sample = self.categorical(probs=probCpu).sample()
        # sample: (BATCHSIZE)
        return prob, sample


class Attending(nn.Module):
    def __init__(
        self,
        use_tanh=False,
        floatDim=16,
        q_dim=None,
        device="cpu"
    ):
        super(Attending, self).__init__()
        if q_dim is None:
            q_dim = floatDim
        self.Attention = Attention(
            use_tanh=use_tanh,
            floatDim=floatDim,
            q_dim=q_dim,
            device=device
        )

    def forward(self, query, ref, mask=None):
        """
            input:
                query: hidden state of decoder at time i, (1, batchSize, floatDim)
                ref: refercne vector [(1, batch, floatDim)]
                mask: masking for selected node. (seq, batchSize)
            output:
                query: (1, batchSize, floatDim)
        """
        prob, _ = self.Attention.forward(query, ref, mask=mask)
        prob = prob.permute(1, 0).contiguous()
        prob = torch.unsqueeze(prob, dim=-1)
        # prob: (SEQ, BATCHSIZE, 1)
        # ref = torch.cat(ref, dim=0)
        # ref: (SEQ, BATCHSIZE, FLOATDIM)
        query = (prob * ref).sum(0)
        return query


class Player:
    def __init__(
        self,
        cfg: NeuralCOCfg,
        logMode=True
    ):
        self._cfg = cfg
        self.device = torch.device(self._cfg.actorDevice)
        self.logMode = logMode
        self.buildModel()
        self.buildOptim()

        if self._cfg.lPath:
            self.loadModel()

        # self._connect = redis.StrictRedis(self._cfg.hostName)
        # for key in self._connect.scan_iter():
        #     self._connect.delete(key)
        self.countModel = 0
        self.env = Euclidean_2D_TSP_Env(
            self._cfg.nodeNum, resolution=self._cfg.resolution, widHei=self._cfg.widHei)
        if self.logMode:
            self.logger = logging.getLogger("CPO")
            self.logger.setLevel(logging.INFO)
            time = datetime.now()
            strTime = time.strftime("%m_%d_%Y_%H_%M")
            self._cfg.loggingPath = os.path.join(
                self._cfg.loggingPath, strTime)
            os.mkdir(self._cfg.loggingPath)
            infoPath = os.path.join(
                self._cfg.loggingPath, "info.log"
            )
            streamPath = os.path.join(
                self._cfg.loggingPath, "data.log"
            )
            evalPath = os.path.join(
                self._cfg.loggingPath, "eval.log"
            )

            savePath = os.path.join(
                self._cfg.loggingPath, "data.pth"
            )

            self._cfg.sPath = savePath

            self.infoLogger = setup_logger("info", infoPath)
            self.dataLogger = setup_logger("data", streamPath)
            self.evalLogger = setup_logger("eval", evalPath)

            data = writeTrainInfo(self._cfg.data)
            self.infoLogger.info(data)
            dataFormat = "step, distance, entropy, norm"
            self.dataLogger.info(dataFormat)
            evalDataFormat = "step, distance"
            self.evalLogger.info(evalDataFormat)

    def buildModel(self):
        for netName, data in self._cfg.agent.items():
            if netName == "Encoder":
                self.Encoder = baseAgent(data)
                # self.Decoder = baseAgent(data)
                self.GEncoder = baseAgent(data)

                self.Encoder.to(self.device)
                # self.Decoder.to(self.device)
                self.GEncoder.to(self.device)
            elif netName == "GDecoder":
                self.GDecoder = baseAgent(data)
                self.GDecoder.to(self.device)
            elif netName == "Decoder":
                self.Decoder = baseAgent(data)
                self.Decoder.to(self.device)
            elif netName == "Embedding":
                self.criticEmbedding = baseAgent(data)
                self.Embedding = baseAgent(data)
                self.criticEmbedding.to(self.device)
                self.Embedding.to(self.device)
        # critic
        self.Attending = Attending(
            floatDim=self._cfg.dimension,
            device=self._cfg.actorDevice,
            use_tanh=self._cfg.useTanh,
            q_dim=self._cfg.dimension
        )

        # actor
        self.EAttending = Attending(
            floatDim=self._cfg.dimension, device=self._cfg.actorDevice, use_tanh=self._cfg.useTanh
        )
        self.Attention = Attention(
            floatDim=self._cfg.dimension, device=self._cfg.actorDevice)
        self.initialEmbedding = torch.nn.Parameter(
            torch.randn(1, 1, self._cfg.dimension).to(self.device))
        self.initialEmbedding.data.uniform_(-(1. / math.sqrt(self._cfg.dimension)),
                                            1. / math.sqrt(self._cfg.dimension))

        self.actorEncoderInit = torch.rand(
            (1, 2, self._cfg.dimension)
        )
        self.actorEncoderInit.data.uniform_(
            -(1. / math.sqrt(self._cfg.dimension)),
            1. / math.sqrt(self._cfg.dimension)
        )
        self.actorEncoderInit = nn.parameter.Parameter(self.actorEncoderInit)
        self.criticEncoderInit = torch.rand(
            (1, 2, self._cfg.dimension)
        )
        self.criticEncoderInit.data.uniform_(
            -(1. / math.sqrt(self._cfg.dimension)),
            1. / math.sqrt(self._cfg.dimension)
        )
        self.criticEncoderInit = nn.parameter.Parameter(self.criticEncoderInit)

        if self._cfg.initCellTrain is False:
            self.actorEncoderInit.requires_grad = False
            self.criticEncoderInit.requires_grad = False

        pp = []
        pp += self.Embedding.getParameters()
        pp += self.criticEmbedding.getParameters()
        pp += self.Encoder.getParameters()
        pp += self.Decoder.getParameters()
        pp += self.GEncoder.getParameters()
        pp += self.GDecoder.getParameters()
        pp += list(self.Attending.parameters())
        pp += list(self.Attention.parameters())
        pp += list([self.initialEmbedding])
        pp += list(self.EAttending.parameters())
        for p in pp:
            torch.nn.init.uniform_(p, -self._cfg.size, self._cfg.size)

    def buildOptim(self):
        for optimName, data in self._cfg.optim.items():
            if optimName == "Actor":
                self.actorOptim = getOptim(
                    data, (self.Encoder, self.Decoder, self.Attention, self.EAttending, self.Embedding))
                self.initOptim = getOptim(
                    data, self.initialEmbedding, floatV=True)
                if self._cfg.initCellTrain:
                    self.ainitOptim = getOptim(
                        data, self.actorEncoderInit, floatV=True)
                    self.cinitOptim = getOptim(
                        data, self.criticEncoderInit, floatV=True
                    )

            if optimName == "Critic":
                self.criticOptim = getOptim(
                    data, (self.GEncoder, self.Attending, self.GDecoder, self.criticEmbedding))

    def step(self):
        calculateNorm = 0
        if self._cfg.clipNorm:
            norm = self._cfg.norm
            pp = []
            pp += self.Embedding.getParameters()
            pp += self.criticEmbedding.getParameters()
            pp += self.Encoder.getParameters()
            pp += self.Decoder.getParameters()
            pp += self.GEncoder.getParameters()
            pp += self.GDecoder.getParameters()
            pp += list(self.Attending.parameters())
            pp += list(self.Attention.parameters())
            pp += list([self.initialEmbedding])
            pp += list(self.EAttending.parameters())
            if self._cfg.initCellTrain:
                pp += list([self.criticEncoderInit])
                pp += list([self.actorEncoderInit])
            for p in pp:
                calculateNorm += p.grad.data.norm(2)
            calculateNorm = calculateNorm.pow(0.5)
            torch.nn.utils.clip_grad_norm_(pp, norm)

        self.actorOptim.step()
        self.initOptim.step()
        self.criticOptim.step()
        if self._cfg.initCellTrain:
            self.ainitOptim.step()
            self.cinitOptim.step()

        return calculateNorm

    def zeroGrad(self):
        self.actorOptim.zero_grad()
        self.initOptim.zero_grad()
        self.criticOptim.zero_grad()
        if self._cfg.initCellTrain:
            self.ainitOptim.zero_grad()
            self.cinitOptim.zero_grad()

    def calculateLoss(self, logProb, critic, distance):
        # -distance: return
        distance = distance.view(-1, 1).detach()
        logProb = logProb.view(-1, 1)
        criticDetach = critic.detach()
        if self._cfg.sum:
            actorLoss = ((distance - criticDetach) *
                         logProb + self._cfg.entropyCoef * logProb).sum()
            criticLoss = ((distance - critic) ** 2).sum()

        actorLoss = ((distance - criticDetach) *
                     logProb + self._cfg.entropyCoef * logProb).mean()
        criticLoss = ((distance - critic) ** 2).mean()
        entropy = (-logProb).mean()
        return actorLoss, criticLoss, entropy

    def loadModel(self):
        modelDict = torch.load(self._cfg.lPath, map_location=self.device)
        self.Embedding = modelDict["Embedding"]
        self.criticEmbedding = modelDict["CriticEmbedding"]

        self.Encoder.loadParameters()
        self.Encoder.load_state_dict(modelDict['Encoder'])

        self.Decoder.loadParameters()
        self.Decoder.load_state_dict(modelDict['Decoder'])

        self.GEncoder.loadParameters()
        self.GEncoder.load_state_dict(modelDict['GEncoder'])

        self.Attending.load_state_dict(modelDict['Attending'])
        self.Attention.load_state_dict(modelDict['Attention'])

        # self.GDecoder.loadParameters()
        self.GDecoder.load_state_dict(modelDict["GDecoder"])

        self.EAttending.load_state_dict(modelDict["EAttending"])
        self.initialEmbedding = modelDict['initalEmbedding']
        # self.actorOptim.load_state_dict(modelDict["actorOptim"])
        # self.criticOptim.load_state_dict(modelDict["criticOptim"])
        self.actorEncoderInit = modelDict['actorEncoderInit']
        self.criticEncoderInit = modelDict['criticEncoderInit']

    def _forward(self, s, t=1):
        PI = torch.zeros((0, self._cfg.batchSize))
        mask = torch.ones(
            (self._cfg.nodeNum, self._cfg.batchSize)).to(self.device)
        s = s.view(-1, 2, 1)
        embedded_state = self.Embedding.forward([s])[0]
        critic_embedded_state = self.criticEmbedding.forward([s])[0]
        embedded_state = embedded_state.view(
            self._cfg.nodeNum, self._cfg.batchSize, -1)
        critic_embedded_state = critic_embedded_state.view(
            self._cfg.nodeNum, self._cfg.batchSize, -1)
        s = s.view(self._cfg.nodeNum, self._cfg.batchSize, 2)
        _s = s.permute(1, 0, 2).contiguous()
        # SEQ*BATCHSIZE, 2

        if self._cfg.zeroCellInit:
            self.Encoder.zeroCellState(self._cfg.batchSize)
            self.Encoder.detachCellState()
            self.GEncoder.zeroCellState(self._cfg.batchSize)
            self.GEncoder.detachCellState()
        else:
            initActorEncoderStates = torch.cat(
                [self.actorEncoderInit[:, 0:1, :] for _ in range(self._cfg.batchSize)], dim=1).to(self.device)
            initCriticEncoderStates = torch.cat(
                [self.criticEncoderInit[:, 0:1, :] for _ in range(self._cfg.batchSize)], dim=1).to(self.device)
            initCellActorState = torch.cat(
                [self.actorEncoderInit[:, 1:, :]
                    for _ in range(self._cfg.batchSize)], dim=1
            ).to(self.device)
            initCellCriticState = torch.cat(
                [self.criticEncoderInit[:, 1:, :]
                    for _ in range(self._cfg.batchSize)], dim=1
            ).to(self.device)
            self.Encoder.setCellState(
                (initActorEncoderStates, initCellActorState))
            self.GEncoder.setCellState(
                (initCriticEncoderStates, initCellCriticState))

        # ref = []
        # gRef = []
        # for i in range(self._cfg.nodeNum):
        #     self.Encoder.forward([embedded_state[i:i+1, :, :]])
        #     self.GEncoder.forward([critic_embedded_state[i:i + 1, :, :]])
        #     if (i + 1) % self._cfg.detachInterval == 0 and self._cfg.detachCellState:
        #         self.Encoder.detachCellState()
        #         self.GEncoder.detachCellState()
        #     ref.append(
        #         self.Encoder.getCellState()[1]
        #     )
        #     gRef.append(
        #         self.GEncoder.getCellState()[1]
        #     )
        # ref = torch.cat(ref, dim=0)
        # gRef = torch.cat(gRef, dim=0)

        ref = self.Encoder.forward([embedded_state])[0]
        gRef = self.GEncoder.forward([critic_embedded_state])[0]

        embedded_state_ = embedded_state.permute(
            1, 0, 2).contiguous().view(-1, self._cfg.dimension)

        batchInitParameter = torch.cat(
            [self.initialEmbedding for _ in range(self._cfg.batchSize)], dim=1)
        if self._cfg.detachCellState:
            self.Encoder.detachCellState()
        self.Decoder.setCellState(self.Encoder.getCellState())
        self.Decoder.forward([batchInitParameter])
        logSelectedProb = torch.zeros(
            (self._cfg.batchSize)).to(self.device)
        distance = []
        for _ in range(self._cfg.nodeNum - 1):
            tempmask = torch.ones(
                (self._cfg.batchSize, self._cfg.nodeNum, 2)).view(-1)
            if _ == 0 and self._cfg.randomInit:
                selectedNode = [random.randint(0, self._cfg.nodeNum-1)
                                for i in range(self._cfg.batchSize)]
                selectedNode = torch.tensor(selectedNode).cpu().long()
                prob = torch.ones(
                    (self._cfg.batchSize, self._cfg.nodeNum)).float().to(self.device)
            else:

                dec = self.Decoder.getCellState()[0]
                query = self.EAttending.forward(dec, ref, mask)
                query = torch.unsqueeze(query, dim=0)
                query += dec

                prob, selectedNode = self.Attention.forward(
                    query, ref, mask, temperature=t, sampleMode="Sampling")
                # meanTime += time.time() - t
            # selectedNode : [batchSize]
            prob = prob.view(-1)
            # prob: (batchSize *ITER, seq)

            # update mask to ignore already selected node.
            mask = mask.permute(1, 0).contiguous().view(-1)
            # seq * batch -> batch * seq
            # mask: (320), (batchSize * seq)
            # tempmask batchSize, seq, 2
            # decodingmask, batch*Iter, seq
            selectedNode = selectedNode.cpu()
            selectedNode_mask = [selectedNode[j].data.item() + j *
                                 self._cfg.nodeNum for j in range(self._cfg.batchSize)]
            selectedTempMask_x = [selectedNode[j].data.item(
            ) * 2 + 2 * j * self._cfg.nodeNum for j in range(self._cfg.batchSize)]
            selectedTempMask_y = [1 + selectedNode[j].data.item(
            ) * 2 + 2 * j * self._cfg.nodeNum for j in range(self._cfg.batchSize)]

            mask[selectedNode_mask] = 0
            tempmask[selectedTempMask_x] = 0
            tempmask[selectedTempMask_y] = 0

            logProb = torch.log(prob[selectedNode_mask])

            # batchSize * seq
            # embedding_state seq, batchSize, dim
            # seq * batch
            selectedEmbeddingState = embedded_state_[selectedNode_mask]
            selectedEmbeddingState = torch.unsqueeze(
                selectedEmbeddingState, dim=0)
            self.Decoder.forward([selectedEmbeddingState])
            if (_ + 1) % 10 == 0 and self._cfg.detachCellState:
                self.Decoder.detachCellState()
            logSelectedProb += logProb
            mask = mask.view(self._cfg.batchSize, self._cfg.nodeNum).permute(
                1, 0).contiguous()
            tempmask = tempmask.view(self._cfg.batchSize, self._cfg.nodeNum, 2)
            # tempmask = tempmask.permute(1, 0, 2).contiguous()
            # SEQ, BATCH, 2
            distance.append(
                _s[tempmask == 0].view(-1, 2)
            )

            # update PI
            # selectedNode: (1, batchSize)
            selectedNode = selectedNode.view(1, -1)
            PI = torch.cat((PI, selectedNode), dim=0)

        totalSum = (self._cfg.nodeNum) * (self._cfg.nodeNum-1)/2
        PI_sum = totalSum - PI.sum(dim=0).view(1, -1)
        PI_sum = PI_sum.view(-1)
        tempmask = torch.ones((self._cfg.batchSize, self._cfg.nodeNum, 2))
        tempmask = tempmask.view(-1)
        selectedTempMask_x = [PI_sum[j].data.item()*2 + 2*j *
                              self._cfg.nodeNum for j in range(self._cfg.batchSize)]
        selectedTempMask_y = [1+PI_sum[j].data.item()*2 + 2*j *
                              self._cfg.nodeNum for j in range(self._cfg.batchSize)]
        tempmask[selectedTempMask_x] = 0
        tempmask[selectedTempMask_y] = 0
        tempmask = tempmask.view(
            self._cfg.batchSize, self._cfg.nodeNum, 2)
        distance.append(
            _s[tempmask == 0].view(-1, 2)
        )
        distance = torch.stack(distance, 0)
        PI_sum = PI_sum.view(1, -1)
        PI = torch.cat((PI, PI_sum), dim=0)
        PI = PI.permute(1, 0).contiguous()

        query = self.GEncoder.getCellState()[0]
        # data = []
        # data.append(query)
        for _ in range(1):
            query = self.Attending.forward(query, gRef)
            query = torch.unsqueeze(query, dim=0)
            # data.append(query)
        # data = torch.cat(data, dim=-1)
        critic = self.GDecoder.forward([query[0]])[0]

        x_pos = distance[:-1, :, 0]
        y_pos = distance[:-1, :, 1]
        x_i_pos = distance[1:, :, 0]
        y_i_pos = distance[1:, :, 1]
        initX = distance[0:1, :, 0]
        initY = distance[0:1, :, 1]
        fX = distance[-1:, :, 0]
        fY = distance[-1:, :, 1]
        d = (((x_pos - x_i_pos) ** 2 + (y_pos - y_i_pos) ** 2).pow(0.5)).sum(0)
        d += ((initX - fX) ** 2 + (initY - fY) ** 2).pow(0.5).sum(0)

        return PI, critic, d, logSelectedProb

    def forward(self, s, ITER=1, sampleMode="Greedy", temp=1):
        # input: sequence data, tuple
        # output: solution of sequence
        # e.g. number of points : 10
        # e.g. batch size :32
        # s: (10, 32, 2), <seq, batch, dim>
        PI = torch.zeros((0, self._cfg.batchSize*ITER))
        sT = s.permute(1, 0, 2).contiguous()
        sT = torch.cat([sT for _ in range(ITER)], dim=0)

        # --------------------- Encoder ---------------------------

        # mask: (10, 32)
        mask = torch.ones(
            (self._cfg.nodeNum, self._cfg.batchSize*ITER)).to(self.device)

        # embedding = self.Embedding.repeat(self._cfg.batchSize, 1, 1)
        # embedded_state = torch.matmul(s, self.Embedding)
        # critic_embedded_state = torch.matmul(s, self.criticEmbedding)

        s = s.view(-1, 2, 1)
        embedded_state = self.Embedding.forward([s])[0]
        critic_embedded_state = self.criticEmbedding.forward([s])[0]
        embedded_state = embedded_state.view(
            self._cfg.nodeNum, self._cfg.batchSize, -1)
        critic_embedded_state = critic_embedded_state.view(
            self._cfg.nodeNum, self._cfg.batchSize, -1)
        s = s.view(self._cfg.nodeNum, self._cfg.batchSize, 2)
        # _s = s.permute(1, 0, 2).contiguous()

        embedded_state = torch.cat(
            [embedded_state for _ in range(ITER)], dim=1)
        critic_embedded_state = torch.cat(
            [critic_embedded_state for _ in range(ITER)], dim=1)
        # SEQ, BATCH*ITER, 128, -> BATCH1, BATCH2, ---
        if self._cfg.zeroCellInit:
            self.Encoder.zeroCellState(self._cfg.batchSize * ITER)
            self.Encoder.detachCellState()
            self.GEncoder.zeroCellState(self._cfg.batchSize * ITER)
            self.GEncoder.detachCellState()
        else:
            initActorEncoderStates = torch.cat(
                [self.actorEncoderInit[:, 0:1, :] for _ in range(ITER*self._cfg.batchSize)], dim=1).to(self.device)
            initCriticEncoderStates = torch.cat(
                [self.criticEncoderInit[:, 0:1, :] for _ in range(ITER * self._cfg.batchSize)], dim=1).to(self.device)

            initCellActorState = torch.cat(
                [self.actorEncoderInit[:, 1:, :]
                    for _ in range(ITER * self._cfg.batchSize)], dim=1
            ).to(self.device)
            initCellCriticState = torch.cat(
                [self.criticEncoderInit[:, 1:, :]
                    for _ in range(ITER * self._cfg.batchSize)], dim=1
            ).to(self.device)
            self.Encoder.setCellState(
                (initActorEncoderStates, initCellActorState))
            self.GEncoder.setCellState(
                (initCriticEncoderStates, initCellCriticState))

        ref = self.Encoder.forward([embedded_state])[0]
        gRef = self.GEncoder.forward([critic_embedded_state])[0]
        # ref = []
        # gRef = []
        # for i in range(self._cfg.nodeNum):
        #     self.Encoder.forward([embedded_state[i:i+1]])
        #     self.GEncoder.forward([critic_embedded_state[i:i+1]])
        #     ref.append(
        #         self.Encoder.getCellState()[1]
        #     )
        #     gRef.append(
        #         self.GEncoder.getCellState()[1]
        #     )
        # ref = torch.cat(ref, dim=0)
        # gRef = torch.cat(gRef, dim=0)

        # embedded_state_test = embedded_state.clone()
        embedded_state = embedded_state.permute(1, 0, 2).contiguous()
        # batch, seq, dim
        embedded_state = embedded_state.view(-1, self._cfg.dimension)
        # --------------------------------------------------------

        # -------------------- Decoder -----------------------------
        batchInitParameter = torch.cat(
            [self.initialEmbedding for _ in range(self._cfg.batchSize * ITER)], dim=1)
        self.Decoder.setCellState(self.Encoder.getCellState())
        self.Decoder.forward([batchInitParameter])
        logSelectedProb = torch.zeros(
            (self._cfg.batchSize * ITER)).to(self.device)
        # logSelectedProb: (batchSize)
        distance = []
        for _ in range(self._cfg.nodeNum - 1):
            tempmask = torch.ones(
                (self._cfg.batchSize * ITER, self._cfg.nodeNum, 2)).view(-1)

            decodingMask = torch.ones(
                (self._cfg.batchSize * ITER, self._cfg.nodeNum)
            )
            if _ == 0 and self._cfg.randomInit:
                selectedNode = [random.randint(0, self._cfg.nodeNum-1)
                                for i in range(self._cfg.batchSize*ITER)]
                selectedNode = torch.tensor(selectedNode).float()
                prob = torch.ones(
                    (self._cfg.batchSize*ITER, self._cfg.nodeNum)).float().to(self.device)
            else:

                dec = self.Decoder.getCellState()[0]
                query = self.EAttending.forward(dec, ref, mask)
                query = torch.unsqueeze(query, dim=0)
                query += dec
                if sampleMode == "Greedy":
                    prob, selectedNode = self.Attention.forward(
                        query, ref, mask, temperature=temp, sampleMode="Greedy")
                else:
                    prob, selectedNode = self.Attention.forward(
                        query, ref, mask, temperature=temp, sampleMode=sampleMode)
            # prob, selectedNode = self.Attention.forward(
            #     query, ref, mask, temperature=2, sampleMode="Sampling")

            prob = prob.view(-1)
            # prob: (batchSize *ITER, seq)

            # update mask to ignore already selected node.
            mask = mask.permute(1, 0).contiguous().view(-1)
            # seq * batch -> batch * seq

            decodingMask = decodingMask.view(-1)
            tempmask = tempmask.view(-1)
            # mask: (320), (batchSize * seq)
            # tempmask batchSize, seq, 2
            # decodingmask, batch*Iter, seq

            selectedNode_mask = [selectedNode[j].data.item() + j *
                                 self._cfg.nodeNum for j in range(self._cfg.batchSize*ITER)]
            selectedTempMask_x = [selectedNode[j].data.item() * 2 + 2*j *
                                  self._cfg.nodeNum for j in range(self._cfg.batchSize*ITER)]
            selectedTempMask_y = [1 + selectedNode[j].data.item()*2 + 2*j *
                                  self._cfg.nodeNum for j in range(self._cfg.batchSize*ITER)]
            mask[selectedNode_mask] = 0
            decodingMask[selectedNode_mask] = 0
            tempmask[selectedTempMask_x] = 0
            tempmask[selectedTempMask_y] = 0

            logProb = torch.log(prob[selectedNode_mask])
            decodingMask = decodingMask.view(
                self._cfg.batchSize * ITER, self._cfg.nodeNum).permute(1, 0).contiguous()
            # batchSize * seq
            # embedding_state seq, batchSize, dim
            # seq * batch
            selectedEmbeddingState = embedded_state[selectedNode_mask]
            # batch, dim
            selectedEmbeddingState = torch.unsqueeze(
                selectedEmbeddingState, dim=0)
            self.Decoder.forward([selectedEmbeddingState])
            logSelectedProb += logProb
            mask = mask.view(self._cfg.batchSize*ITER, self._cfg.nodeNum).permute(
                1, 0).contiguous()
            tempmask = tempmask.view(
                self._cfg.batchSize*ITER, self._cfg.nodeNum, 2)
            distance.append(
                sT[tempmask == 0].view(-1, 2)
            )

            # update PI
            # selectedNode: (1, batchSize)
            selectedNode = selectedNode.view(1, -1)
            PI = torch.cat((PI, selectedNode), dim=0)
        totalSum = (self._cfg.nodeNum) * (self._cfg.nodeNum-1)/2
        PI_sum = totalSum - PI.sum(dim=0).view(1, -1)
        PI_sum = PI_sum.view(-1)
        tempmask = torch.ones((self._cfg.batchSize*ITER, self._cfg.nodeNum, 2))
        tempmask = tempmask.view(-1)
        selectedTempMask_x = [PI_sum[j].data.item()*2 + 2*j *
                              self._cfg.nodeNum for j in range(self._cfg.batchSize*ITER)]
        selectedTempMask_y = [1+PI_sum[j].data.item()*2 + 2*j *
                              self._cfg.nodeNum for j in range(self._cfg.batchSize*ITER)]
        tempmask[selectedTempMask_x] = 0
        tempmask[selectedTempMask_y] = 0
        tempmask = tempmask.view(
            self._cfg.batchSize*ITER, self._cfg.nodeNum, 2)
        distance.append(
            sT[tempmask == 0].view(-1, 2)
        )
        distance = torch.stack(distance, 0)
        PI_sum = PI_sum.view(1, -1)
        PI = torch.cat((PI, PI_sum), dim=0)
        PI = PI.permute(1, 0).contiguous()

        # -------------------- Critic -----------------------------

        # To get referecen vector, iteration

        query = self.GEncoder.getCellState()[0]
        # data = []
        # data.append(query)
        for _ in range(1):
            query = self.Attending.forward(query, gRef)
            query = torch.unsqueeze(query, dim=0)
            # data.append(query)
        # data = torch.cat(data, dim=-1)
        critic = self.GDecoder.forward([query[0]])[0]
        # ---------------------------------------------------------

        # -------------------- Distance ---------------------------
        # s: (SEQ, batchSize, 2)
        # PI: (batchSize, SEQ)
        x_pos = distance[:-1, :, 0]
        y_pos = distance[:-1, :, 1]
        x_i_pos = distance[1:, :, 0]
        y_i_pos = distance[1:, :, 1]
        initX = distance[0:1, :, 0]
        initY = distance[0:1, :, 1]
        fX = distance[-1:, :, 0]
        fY = distance[-1:, :, 1]
        # d = (((x_pos - x_i_pos) ** 2 + (y_pos - y_i_pos) ** 2 +
        #       (initX - fX) ** 2 + (initY - fY) ** 2).pow(0.5)).sum(0)
        d = (((x_pos - x_i_pos) ** 2 + (y_pos - y_i_pos) ** 2).pow(0.5)).sum(0)
        d += ((initX - fX) ** 2 + (initY - fY) ** 2).pow(0.5).sum(0)
        dd = d.view(ITER, self._cfg.batchSize)
        dd = dd.permute(1, 0).contiguous()
        # batchSize, ITER
        mmask = torch.ones((self._cfg.batchSize*ITER))
        minD = dd.min(1)[1]
        minD += torch.tensor([i * ITER for i in range(self._cfg.batchSize)]
                             ).long().to(self.device)
        mmask[minD] = 0
        mmask = mmask.view(self._cfg.batchSize, ITER)
        mmask = mmask.permute(1, 0).contiguous()
        dd = dd.permute(1, 0).contiguous()
        critic = critic.view(ITER, self._cfg.batchSize)
        PI = PI.view(ITER, self._cfg.batchSize, -1)
        logSelectedProb = logSelectedProb.view(ITER, self._cfg.batchSize)
        xPI = PI[mmask == 0]
        xLogSelectedProb = logSelectedProb[mmask == 0]
        xDD = dd[mmask == 0]
        xCritic = critic[mmask == 0]
        xCritic = xCritic.unsqueeze(1)

        return xPI, xCritic, xDD, xLogSelectedProb

    def run(self, testSize=50):

        testSequences = []
        BATCHSIZE = self._cfg.batchSize
        for i in range(testSize):
            _, data = self.env.generateRandom()
            testSequences.append(torch.tensor(
                data["xy_pos"]).float().to(self.device))
        testSequences = torch.stack(testSequences, dim=1)
        for t in count():

            sequences = []
            for i in range(self._cfg.batchSize):
                _, data = self.env.generateRandom()
                sequences.append(torch.tensor(
                    data["xy_pos"]).float().to(self.device))
            sequences = torch.stack(sequences, dim=1)
            PI, critic, distance, logProb = self._forward(
                sequences, t=1)
            actorLoss, criticLoss, entropy = self.calculateLoss(
                logProb, critic, distance)
            loss = actorLoss + criticLoss
            loss.backward()
            norm = self.step()
            self.zeroGrad()

            d = distance.mean().detach().cpu().numpy()
            e = entropy.detach().cpu().numpy()
            n = norm.detach().cpu().numpy()
            if self.logMode:
                self.dataLogger.info(
                    "{}, {:.3f}, {:.3f}, {:.3f}".format(t+1, d, e, n)
                )
            if (t + 1) % 1000 == 0:
                with torch.no_grad():
                    totalDistance = 0
                    self.Mode(False)
                    for j in range(testSize):
                        self._cfg.batchSize = 1
                        PI, critic, distance, logProb = self.forward(
                            testSequences[:, j:j+1, :], ITER=128, sampleMode="Sampling", temp=2)
                        totalDistance += distance
                if self.logMode:
                    self.evalLogger.info(
                        "{}, {:.3f}".format(
                            t+1, (totalDistance / testSize).detach().cpu().numpy()[0])
                    )
                self.Mode()
                self._cfg.batchSize = BATCHSIZE
            if (t+1) % self._cfg.decayStep == 0:
                for g in self.actorOptim.param_groups:
                    g["lr"] *= self._cfg.decay
                for g in self.criticOptim.param_groups:
                    g["lr"] *= self._cfg.decay
                model = {}
                model["Embedding"] = self.Embedding
                model["Encoder"] = self.Encoder.state_dict()
                model["Decoder"] = self.Decoder.state_dict()
                model["GEncoder"] = self.GEncoder.state_dict()
                model['Attending'] = self.Attending.state_dict()
                model['Attention'] = self.Attention.state_dict()
                model['GDecoder'] = self.GDecoder.state_dict()

                model['EAttending'] = self.EAttending.state_dict()

                model['initalEmbedding'] = self.initialEmbedding
                model['actorOptim'] = self.actorOptim.state_dict()
                model['criticOptim'] = self.criticOptim.state_dict()
                model['actorEncoderInit'] = self.actorEncoderInit
                model['criticEncoderInit'] = self.criticEncoderInit
                model['CriticEmbedding'] = self.criticEmbedding
                torch.save(model, self._cfg.sPath)

    def Mode(self, cond=True):
        self.Embedding.train(cond)
        self.criticEmbedding.train(cond)

    def eval(self):
        iter = self._cfg.eval['iter']
        temp = self._cfg.eval['temperature']
        samplingMode = self._cfg.eval['samplingMode']
        if samplingMode == "Greedy":
            plotMode = False
        else:
            plotMode = self._cfg.eval['plot']
        testSize = self._cfg.eval['testSize']
        testSequences = []
        totlaDistance = 0

        self.Mode(False)

        for i in range(testSize):
            # _, data = self.env.generateRandomDistanceMatrix()
            _, data = self.env.generateRandom()
            testSequences.append(torch.tensor(
                data["xy_pos"]).float().to(self.device))
        testSequences = torch.stack(testSequences, dim=1)
        with torch.no_grad():
            if samplingMode == "Sampling":
                self._cfg.batchSize = 1
                count = testSize
                interval = 1
            elif samplingMode == "Greedy":
                self._cfg.batchSize = testSize
                count = 1
                interval = testSize
            else:
                RuntimeError("You can specify only Greedy or Sampling.")
            for j in range(count):
                PI, _, distance, __ = self.forward(
                    testSequences[:, j:j+interval,
                                  :], ITER=iter, sampleMode=samplingMode, temp=temp
                )
                totlaDistance += distance.mean()
                if plotMode:
                    title = 'TSP'+str(self._cfg.nodeNum)
                    self.env.plotLine(
                        testSequences[:, j:j+1, :].cpu(), PI.cpu(), widHei=self._cfg.widHei, title=title)
        if samplingMode == "Greedy":
            testSize = 1
        print("Mean Distance : {:.3f}".format(
            totlaDistance.detach().cpu().numpy()/testSize))

    @staticmethod
    def plot(cfg):
        path = cfg.plot_path
        dataLogPath = os.path.join(
            path, "data.log"
        )
        evalLogPath = os.path.join(
            path, "eval.log"
        )
        data = pd.read_csv(dataLogPath)
        columns = data.columns
        data = data.to_numpy()
        step = data[:, 0]

        data_eval = pd.read_csv(evalLogPath)
        data_eval = data_eval.to_numpy()
        step_eval = data_eval[:, 0]

        plotSize = data.shape[1]
        plt.suptitle(path)
        for i in range(plotSize-1):
            plt.subplot(1, plotSize, i+1)
            plt.plot(step, data[:, i+1])
            plt.xscale("log")
            plt.xlabel("Step : [Log Scale]")
            plt.ylabel(columns[i+1])
            plt.title(columns[i+1])
        plt.subplot(1, plotSize, plotSize)
        plt.plot(step_eval, data_eval[:, 1])
        minimum = np.min(data_eval[:, 1])
        plt.xscale("log")
        plt.xlabel("Step : [Log Scale]")
        plt.ylabel("Eval Distance")
        plt.title("Evaluation:{:.2f}".format(minimum))
        plt.show()
