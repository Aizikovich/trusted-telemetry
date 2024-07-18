import torch.nn as nn
import torch


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               dropout=dropout, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=num_layers,
                               dropout=dropout, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(hidden_size*2, input_size)  # Added linear layer to match input size

    def forward(self, x):
        x, _ = self.encoder(x)
        x, _ = self.decoder(x)
        x = self.output_layer(x)  # Map the hidden states back to the input feature space
        return x


class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cell_id, dropout=0.1):
        super(LSTMDecoder, self).__init__()
        self.cell_id = cell_id if isinstance(cell_id, str) else str(cell_id)
        self.hidden_size = hidden_size
        # input is latent space of cell (1, 3, 256) cat with avg of other cells (1, 3, 256) overall (1, 3, 512)
        self.decoder = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=num_layers,
                               dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, input_size)  # Added linear layer to match input size

    def make_input(self, embeddings):
        x = embeddings[self.cell_id]
        other_tensors = [tensor for k, tensor in embeddings.items() if k != self.cell_id]
        avg_tensor = torch.mean(torch.stack(other_tensors), dim=0)
        x = torch.cat((x, avg_tensor), dim=-1)
        return x.squeeze(0)

    def forward(self, x):
        x = self.make_input(x)
        x, _ = self.decoder(x)
        x = self.fc(x)
        x = self.output_layer(x)  # Map the hidden states back to the input feature space
        return x
