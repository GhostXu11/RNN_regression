import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from tqdm import tqdm

data = pd.read_csv('./data/AirPassengers_new.csv')
data = data.iloc[:, 2:5].values.astype(float)

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(data)

scaler_y = MinMaxScaler(feature_range=(-1, 1))
scaler_y.fit(data[:, 2].reshape(-1, 1))

train_data_length = len(train_data_normalized)
train_data_normalized, val_data_normalized = train_data_normalized[:int(0.7 * train_data_length)], \
                                             train_data_normalized[int(0.7 * train_data_length) - 24:]

print(len(val_data_normalized))
train_window = 20
batch_size = 12
input_dim = 3
hidden_dim = 200
layer_dim = 2
output_dim = 1


def create_inout_sequences(input_data, tw):
    seq = []
    label = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1, 1]
        seq.append(train_seq)
        label.append(train_label)
    return seq, label


train_seq, train_label = create_inout_sequences(train_data_normalized, train_window)
val_seq, val_label = create_inout_sequences(val_data_normalized, train_window)

train_seq = torch.from_numpy(np.array(train_seq)).float()
train_label = torch.from_numpy(np.array(train_label)).float()
val_seq = torch.from_numpy(np.array(val_seq)).float()
val_label = torch.from_numpy(np.array(val_label)).float()
print(val_seq.shape)
print(val_label.shape)

# print(train_seq.shape)
train_dataset = TensorDataset(train_seq, train_label)
val_dataset = TensorDataset(val_seq, val_label)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,
                            batch_first=True)  # batch_first=True (batch_dim, seq_dim, feature_dim)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])

        return out


def train_LSTM(epochs):
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    min_valid_loss = np.inf

    epochs = epochs

    for epoch in range(epochs):
        # train start
        train_loss = 0.0
        for ids, (seq, labels) in enumerate(trainloader):
            optimizer.zero_grad()

            y_pred = model(seq)
            # labels = labels.squeeze(dim=1)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            train_loss += single_loss.item()

        # valid start
        valid_loss = 0.0
        model.eval()
        for ids, (seq, labels) in enumerate(validloader):
            optimizer.zero_grad()

            y_pred = model(seq)
            # labels = labels.squeeze(dim=1)
            single_loss = loss_function(y_pred, labels)
            valid_loss += single_loss.item()
        print(
            f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t '
            f'Validation Loss: {valid_loss / len(validloader)}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), './checkpoint/saved_model.pth')


def test_LSTM():
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    PATH = './checkpoint/saved_model.pth'
    model.load_state_dict(torch.load(PATH))
    model.eval()
    y_true = []
    predicted = []
    for ids, (seq, labels) in enumerate(validloader):
        y_pred = model(seq)
        labels = labels.squeeze(dim=1)
        predicted.append(y_pred)
        y_true.append(labels)
    return y_true, predicted


def plot_results(predicted_data, true_data, filename):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label="True Data")
    plt.plot(predicted_data, label="Prediction")
    plt.legend()
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    # train_LSTM(300)
    test, pred = test_LSTM()
    prediction = []
    true = []
    for i in pred:
        res = i.detach().numpy()
        for j in res:
            prediction.append(j[0])
    for i in test:
        res = i.detach().numpy()
        for j in res:
            true.append(j)

    prediction = np.array(prediction)
    prediction = prediction.reshape(-1, 1)
    prediction = scaler_y.inverse_transform(prediction)

    true = np.array(true)
    true = true.reshape(-1, 1)
    true = scaler_y.inverse_transform(true)
    plot_results(prediction, true, 'res.jpg')
