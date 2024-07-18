# from torch.utils.tensorboard import SummaryWriter
import math
from collections import defaultdict
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from farmeworks.framework import Framework


class AE(Framework):
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter(log_dir='../tensorboard')
        self.metrics_ae = defaultdict(list)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def train_single_ae_cell(self, dataloader, epochs, name='cell_0'):
        metrics_ae = defaultdict(list)
        self.model.to(self.device)
        optimizer = self.optimizer
        criterion = self.criterion
        self.model.train()
        memo = set()
        start = time.time()
        for epoch in range(epochs):
            ep_start = time.time()
            running_loss = 0.0
            for bx, data in enumerate(dataloader):
                data = data.to(self.device)
                # print(f"in train_single_ae_cell: {data.shape}")
                try:
                    sample = self.model(data)
                except RuntimeError as e:
                    print(f"bx: {bx}")
                    print(f"data: {data}")
                    memo.add(bx)
                    break
                    # continue
                loss = criterion(sample, data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if math.isnan(loss.item()):
                    print(f"samples: {sample}, \ndata: {data}")
                running_loss += loss.item()
                # print(f"loss: {loss.item()}, type: {type(loss.item())}")
            epoch_loss = running_loss / len(dataloader)
            print(running_loss)
            self.writer.add_scalar(f'Loss/train/{name}_{epochs}', epoch_loss, epoch)
            metrics_ae['train_loss'].append(epoch_loss)
            ep_end = time.time()
            if epoch % 10 == 0 or epoch == epochs - 1:
                print('-----------------------------------------------')
                print('[EPOCH] {}/{}\n[LOSS] {}'.format(epoch + 1, epochs, epoch_loss))
                print(f'Epoch Complete in {ep_end - ep_start}')
        self.metrics_ae = metrics_ae
        end = time.time()
        print('-----------------------------------------------')
        print('[System Complete: {}m]'.format((end - start) / 60))
        print(f"epoch: {epochs}, num of untrainable samples: {len(memo) / len(dataloader)}")
        self.save_model(path=f'models/singel_cell_ae/{name}.pth')

    def evaluate_single_ae_cell(self, normal_loader, anomaly_loader, name='cell_0'):
        loss_dist_normal = []
        loss_dist_anomaly = []

        memo = set()
        criterion = self.criterion

        self.model.eval()
        with torch.no_grad():
            for bx, data in enumerate(normal_loader):
                data.to(self.device)
                try:
                    sample = self.model(data)
                except RuntimeError as e:
                    memo.add(bx)
                    continue
                loss = criterion(sample, data)
                loss_dist_normal.append(loss.item())

            for bx, data in enumerate(anomaly_loader):
                data.to(self.device)
                try:
                    sample = self.model(data)
                except RuntimeError as e:
                    memo.add(bx)
                    continue
                loss = criterion(sample, data)
                loss_dist_anomaly.append(loss.item())
        return loss_dist_normal, loss_dist_anomaly