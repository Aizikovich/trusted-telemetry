from collections import defaultdict

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import time
import math

from datasets import DatasetDecoder


class Decoder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter(log_dir='../tensorboard')
        self.metrics_decoder = defaultdict(list)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def train_single_decoder_cell(self, dataloader, epochs, name='decoder'):
        metrics_decoder = defaultdict(list)
        self.model.to(self.device)
        optimizer = self.optimizer
        criterion = self.criterion
        self.model.train()
        memo = set()
        start = time.time()
        for epoch in range(epochs):
            ep_start = time.time()
            running_loss = 0.0
            for bx, (data, label) in enumerate(dataloader):
                data = data
                try:
                    sample = self.model(data)
                except RuntimeError as e:
                    memo.add(bx)
                    continue

                loss = criterion(sample, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if math.isnan(loss.item()):
                    print(f"samples: {sample}, \ndata: {data}")
                running_loss += loss.item()
            epoch_loss = running_loss / len(dataloader)
            self.writer.add_scalar(f'Loss/train/{name}_{epochs}', epoch_loss, epoch)
            metrics_decoder['train_loss'].append(epoch_loss)
            ep_end = time.time()
            if epoch % 10 == 0 or epoch == epochs - 1:
                print('-----------------------------------------------')
                print('[EPOCH] {}/{}\n[LOSS] {}'.format(epoch + 1, epochs, epoch_loss))
                print(f'Epoch Complete in {ep_end - ep_start}')
        self.metrics_ae = metrics_decoder
        end = time.time()
        print('-----------------------------------------------')
        print('[System Complete: {}m]'.format((end - start) / 60))
        print(f"epoch: {epochs}, num of untrainable samples: {len(memo) / len(dataloader)}")
        self.save_model(path=f'models/{name}.pth')

    def evaluate_decoder_cell(self, latent_space, cell_id, name='decoder'):
        print(f"evaluating cell_{cell_id}")
        nor_loss_dist = []
        ano_loss_dist = []
        memo = set()
        self.model.eval()
        criterion = self.criterion
        with torch.no_grad():
            for i in range(1, 7):
                cell_set = DatasetDecoder(latent_space, cell_id=i)

                print(f"cell_{i} len: {len(cell_set)}")
                if len(cell_set) == 0:
                    continue
                cell_loader = DataLoader(cell_set, batch_size=1)
                for bx, (data, label) in enumerate(cell_loader):
                    try:
                        sample = self.model(data)
                    except RuntimeError as e:
                        memo.add(bx)
                        # print(f'\nbx {bx} data: {data} \n')
                        continue
                    loss = criterion(sample, label)
                    if loss.item() > 2.18 and i != cell_id:
                        print(f"cell_{i} bx {bx} | loss: {loss.item()}")

                    if i == cell_id:
                        ano_loss_dist.append(loss.item())
                    else:
                        nor_loss_dist.append(loss.item())

        print(f"memo: {memo}")
        print(f"len normal: {len(nor_loss_dist)} len anomaly: {len(ano_loss_dist)}")

        print(f"normal loss: {nor_loss_dist}")
        print(f"anomaly loss: {ano_loss_dist}")
        return nor_loss_dist, ano_loss_dist
