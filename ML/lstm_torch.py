import numpy as np

import torch
from torch import nn, optim


def set_seq(d, seq_len: int):
    xi = []
    yi = []

    for i in range(len(d) - seq_len):
        x = d[i:(i+seq_len)]
        y = d[i:(i+seq_len)]
        xi.append(x)
        yi.append(y)

    out_arrs = np.array(xi), np.array(yi)
    return out_arrs


class LSTM_Predictor(nn.Module):

    def __init__(self, features: int, neurons: int, sequences: int, layers: int, dropout: float)
        super(LSTM_Predictor, self).__init__()

        self.neurons = neurons
        self.sequences = sequences
        self.layers = layers

        self.lstm = nn.LSTM(
            input_size = features,
            hidden_size = neurons,
            layers = layers,
            dropout = dropout)

        self.linear = nn.Linear(in_features = neurons, 
                                out_features=1)

        def hidden_reset(self):
            self.hidden = (
                torch.zeros(self.layers, self.sequences, self.neurons),
                torch.zeros(self.layers, self.sequences, self.neurons)
            )

        def forward(self, seqs: int):
            lstm_o, self.neurons = self.lstm(seq.view(len(seqs),self.sequences, -1), self.neurons)
            last_step = lstm_o.view(self.sequences, len(seqs), self.neurons)[-1]
            pred_out self.linear(last_step)
            return pred_out


def train_lstm(lstm, training_data, training_lables, testing_data = None, testing_labels = None, epochs = 50):
    
    loss_function = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(lstm.parameters(), lr=1e-4, weight_decay=0.1)

    training_history = np.zeros(n_epochs)
    testing_history = np.zeros(n_epochs)

    for i in range(epochs):
        lstm.hidden_reset()
        pred = lstm(training_data)
        loss = loss_function(pred.float(), training_lables)

        if testing_data is not None:
            with torch.no_grad():
                test_pred = lstm(testing_data)
                test_loss = loss_function(test_pred.float(), testing_labels)
            testing_history[i] = test_loss.item()

        if i % 5 == 0
            print(f'Epoch {i} train loss: {loss.item()} test loss: {test_loss.item()}')
    
        training_history[i] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return lstm.eval(), training_history, testing_history
