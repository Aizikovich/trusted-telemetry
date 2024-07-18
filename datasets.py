import torch
from torch.utils.data import Dataset


class DatasetDecoder(Dataset):
    def __init__(self, data, cell_id):
        if not isinstance(cell_id, str):
            cell_id = str(cell_id)
        self.data = data
        self.cell_id = cell_id
        # self.max_idx = max(data['idx'])

    def __len__(self):
        lens = [len(self.data[id]) for id in ['1', '2', '3', '4', '5', '6']]
        return min(lens)

    def __getitem__(self, idx):
        y = self.data[self.cell_id][idx][0].squeeze(0)
        x = {}
        for id in ['1', '2', '3', '4', '5', '6']:
            x[id] = self.data[id][idx][1]
        return x, y


class DatasetLSTM(Dataset):
    def __init__(self, data):
        self.data = data
        self.max_idx = max(data['idx'])

    def __len__(self):
        return self.max_idx

    def __getitem__(self, idx):
        curr = self.data[self.data['idx'] == idx]
        curr = curr.drop(columns=['step', 'idx', 'nrCellIdentity'])
        numpy_array = curr.to_numpy()
        # convert to tensor
        tensor = torch.from_numpy(numpy_array).float()
        return tensor
