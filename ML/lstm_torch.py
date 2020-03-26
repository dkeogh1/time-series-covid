import numpy as np
import pandas as pd

import torch
from torch import nn, optim

from sklearn.preprocessing import MinMaxScaler

class LSTM_Predictor(nn.Module):

    def __init__(self, features, neurons, sequences, layers, dropout):
        super(LSTM_Predictor, self).__init__()

        self.neurons = neurons
        self.sequences = sequences
        self.layers = layers

        self.lstm = nn.LSTM(
            input_size = features,
            hidden_size = neurons,
            num_layers = layers,
            dropout = dropout)

        self.linear = nn.Linear(in_features = neurons, 
                                out_features=1)

    def hidden_reset(self):
        self.hidden = (
            torch.zeros(self.layers, self.sequences, self.neurons),
            torch.zeros(self.layers, self.sequences, self.neurons))

    def forward(self, seqs):
        lstm_out, self.hidden = self.lstm(
        seqs.view(len(seqs), self.sequences, -1),
        self.hidden
        )
        last_time_step = \
        lstm_out.view(self.sequences, len(seqs), self.neurons)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred

class LSTM_data_loader():
    
    def __init__(self, region_abr, region_list, state_mapper, country, df):
        
        self.df = df
        self.region_abr = region_abr
        self.country = country
        self.region_list = region_list
        self.state_mapper = state_mapper
        self.scaler = MinMaxScaler()
        self.train_data = None
        self.test_data = None
        
    def _remap_to_abr(self, state):
        return self.state_mapper[state]
        
    def subset_df(self):
        
        if self.region_abr:
            
            mask1 = self.df['Country/Region'].str.contains(self.country)
            self.df = self.df[mask1]
            
            counter = 0
            for r in self.df['Province/State'].tolist():
                if self.region_abr not in r:
                    try:
                        if self.region_abr == self._remap_to_abr(r):
                            self.df['Province/State'].iloc[counter] = self._remap_to_abr(r)
                    except KeyError:
                        pass
                    else:
                        self.df.drop(self.df.index[counter])
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
            
    def transform_df_datetime(self, delta=False):
        if not delta:
            self.df = self.df.sum(axis=0)
            self.df.index = pd.to_datetime(self.df.index)
            print("Data in cumulative")

        else:
            self.df = self.df.sum(axis=0)
            self.df.index = pd.to_datetime(self.df.index)
            self.df = self.df.diff().fillna(self.df[0]).astype(np.int64)
            print("Data is converted to daily delta")

    def drop_empty_days(self):
        self.df = self.df.loc[:, (self.df != 0).any(axis=0)]
    
    def gen_data_sets(self,test_data_size=0):   
        if test_data_size:
            self.train_data = self.df[:-test_data_size]
            self.scaler = self.scaler.fit(np.expand_dims(self.train_data, axis=1))
            self.train_data = self.scaler.transform(np.expand_dims(self.train_data, axis=1))

            self.test_data = self.df[-test_data_size:]
            self.test_data = self.scaler.transform(np.expand_dims(self.test_data, axis=1))

        else:
            self.train_data = self.df
            self.scaler = self.scaler.fit(np.expand_dims(self.train_data, axis=1))
            self.train_data = self.scaler.transform(np.expand_dims(self.train_data, axis=1))      
        
    def set_seq(self, train, sequence_lenth):
        if train:
            xs = []
            ys = []

            for i in range(len(self.train_data)-sequence_lenth-1):
                x = self.train_data[i:(i+sequence_lenth)]
                y = self.train_data[i+sequence_lenth]
                xs.append(x)
                ys.append(y)

            return np.array(xs), np.array(ys)
        
        else:
            xs = []
            ys = []

            for i in range(len(self.test_data)-sequence_lenth-1):
                x = self.test_data[i:(i+sequence_lenth)]
                y = self.test_data[i+sequence_lenth]
                xs.append(x)
                ys.append(y)

            return np.array(xs), np.array(ys)
          


def train_lstm(lstm, training_data, training_lables, testing_data = None, testing_labels = None, epochs = 50):
    
    loss_function = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(lstm.parameters(), lr=1e-3, weight_decay=0.1)

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

        if i % 40 == 0:
            if testing_data is not None:
                print(f'Epoch {i} train loss: {loss.item()} test loss: {test_loss.item()}')
            else:
                print(f'Epoch {i} train loss: {loss.item()}')
        training_history[i] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return lstm.eval(), training_history, testing_history

def predict_future(n_future, time_data, sequece_lenth, model):
    test_seq = time_data[:1]
    with torch.no_grad():
        preds = []
        for _ in range(n_future):
            y_test_pred = model(test_seq)
            pred = torch.flatten(y_test_pred).item()
            preds.append(pred)
            new_seq = test_seq.numpy().flatten()
            new_seq = np.append(new_seq, [pred])
            new_seq = new_seq[1:]
            test_seq = torch.as_tensor(new_seq).view(1, sequece_lenth, 1).float()

        return preds
