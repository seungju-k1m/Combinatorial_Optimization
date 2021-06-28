from baseline.utils import jsonParser


class Cfg:
    def __init__(self, path):
        parser = jsonParser(path)
        self.data = parser.loadParser()
        self.batchSzie = 128
        self.dimension = 128
        self.nodeNum = 50
        self.norm = 1
        self.sum = False
        self.lPath = None
        self.device = "cpu"
        self.decay = 0.96
        self.decayStep = 5000
        self.loggingPath = "./logs"
        self.agent = {}
        self.optim = {}
        for key, value in self.data.items():
            setattr(self, key, value)
