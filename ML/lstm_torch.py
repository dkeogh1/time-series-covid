import numpy as np
import pandas as pd

import torch
from torch import nn, optim

from sklearn.preprocessing import MinMaxScaler


def set_seq(d, seq_len: int):
    xi = []
    yi = []

    for i in range(len(d) - seq_len):
        x = d[i:(i+seq_len)]
        y = d[i:+seq_len]
        xi.append(x)
        yi.append(y)

    return np.array(xi), np.array(yi)

class LSTM_Predictor(nn.Module):

    def __init__(self, features: int, neurons: int, sequences: int, layers: int, dropout: float):
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
            pred_out = self.linear(last_step)
            return pred_out

class LSTM_data_loader():
    
    def __init__(self, region_abr, region_list, state_mapper, country, df, sequence_lenth):
        
        self.df = df
        self.region_abr = region_abr
        self.country = country
        self.region_list = region_list
        self.state_mapper = state_mapper
        self.scaler = MinMaxScaler()
        self.train_data = None
        self.test_data = None
        self.sequence_lenth = 3
        
    def _remap_to_abr(self, state):
        return self.state_mapper[state]
        
    def subset_df(self):
        
        if self.region_abr:
            
            mask1 = self.df['Country/Region'].str.contains(self.country)
            self.df = self.df[mask1]
            
            for x in self.region_list:
                counter = 0
                for r in self.df['Province/State'].tolist():
                    if x in r:
                        self.df['Province/State'].iloc[counter] = x
                    elif r in self.state_mapper.keys():
                        try:
                            self.df['Province/State'].iloc[counter] = self._remap_to_abr(r)
                        except Exception as e:
                            print(e)
                    counter += 1
                        
            mask2 = self.df['Province/State'].str.contains(self.region_abr)
            self.df = self.df[mask2]
            self.df = self.df.groupby('Province/State').sum().reset_index()
            self.df = self.df.iloc[:,3:]
                
        elif self.country:
            mask1 = self.df['Country/Region'].str.contains(self.country)
            self.df = self.df[mask1]
            self.df = self.df.groupby('Country/Region').sum().reset_index()
            self.df = self.df.iloc[:,3:]

        else:
            self.df = self.df.iloc[:,4:]
            
    def transform_df_datetime(self):
        self.df = self.df.sum(axis=0)
        self.df.index = pd.to_datetime(self.df.index)
        
    def gen_data_sets(self,test_data_size=0):   
        self.train_data = self.df[:-test_data_size]
        self.scaler = self.scaler.fit(np.expand_dims(self.train_data, axis=1))
        self.train_data = self.scaler.transform(np.expand_dims(self.train_data, axis=1))

        if test_data_size:
            self.test_data = self.df[-test_data_size:]
            self.test_data = self.scaler.transform(np.expand_dims(self.test_data, axis=1))
        
    def set_seq(self, train=True):
        if train:
            xs = []
            ys = []

            for i in range(len(self.train_data)-self.sequence_lenth-1):
                x = self.train_data[i:(i+self.sequence_lenth)]
                y = self.train_data[i+self.sequence_lenth]
                xs.append(x)
                ys.append(y)

            return np.array(xs), np.array(ys)
        
        else:
            xs = []
            ys = []

            for i in range(len(self.test_data)-self.sequence_lenth-1):
                x = self.test_data[i:(i+self.sequence_lenth)]
                y = self.test_data[i+self.sequence_lenth]
                xs.append(x)
                ys.append(y)

            return np.array(xs), np.array(ys)
          


def train_lstm(lstm, training_data, training_lables, testing_data = None, testing_labels = None, epochs = 50):
    
    loss_function = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(lstm.parameters(), lr=1e-4, weight_decay=0.1)

    training_history = np.zeros(epochs)
    testing_history = np.zeros(epochs)

    for i in range(epochs):
        lstm.hidden_reset()
        pred = lstm(training_data)
        loss = loss_function(pred.float(), training_lables)

        if testing_data is not None:
            with torch.no_grad():
                test_pred = lstm(testing_data)
                test_loss = loss_function(test_pred.float(), testing_labels)
            testing_history[i] = test_loss.item()

        if i % 5 == 0:
            print(f'Epoch {i} train loss: {loss.item()} test loss: {test_loss.item()}')
    
        training_history[i] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return lstm.eval(), training_history, testing_history
