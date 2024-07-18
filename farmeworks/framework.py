from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import torch

class Framework:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter(log_dir='../tensorboard')
        self.metrics_ae = defaultdict(list)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
