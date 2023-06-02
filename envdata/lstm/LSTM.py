import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F

def pre_df(df):
    df = df.dropna()
    label = df['person']
    df = df.drop('person', axis=1)
    df = df.drop('regdate', axis=1)
    df = df.drop('PIR', axis=1)
    return df, label

def scale_255(array,i):
    # i,j 위치의 요소들을 추출하여 배열 생성
    first_elements = array[:, i]
    # 첫 번째 요소들의 최솟값과 최댓값 계산
    min_value = np.min(first_elements)
    max_value = np.max(first_elements)
    # 스케일링 수행
    scaled_array = (first_elements - min_value) *(100 / (max_value - min_value))
    return scaled_array

#Train Data
df_train = pd.read_csv('modified_train_data.csv', encoding='ISO-8859-1')
df_train, label_train = pre_df(df_train)

for i in range(9):
    df_train.iloc[:, i] = scale_255(df_train.values, i)

#Test Data
df_test = pd.read_csv("modified_test_data.csv", encoding='ISO-8859-1')
df_test, label_test = pre_df(df_test)
for i in range(9):
    df_test.iloc[:,i] = scale_255(df_test.values,i)


df_train = np.expand_dims(df_train, axis=1)
df_test = np.expand_dims(df_test, axis=1)


class LSTM_CNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_CNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.permute(0, 2, 1)  # Conv1d를 위해 차원 변경
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = out.permute(0, 2, 1)  # 다시 LSTM 입력 형태로 변경
        out = out[:, -1, :]  # 마지막 시퀀스의 출력만 사용
        out = self.fc(out)
        return out


# 모델 인스턴스 생성
input_size = 9  # 입력 특성의 크기
hidden_size = 32  # LSTM의 은닉 상태 크기
num_layers = 2  # LSTM 층의 수
output_size = 7  # 출력 크기
model = LSTM_CNN_Model(input_size, hidden_size, num_layers, output_size)
num_epochs = 300
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

best_accuracy = 0.0
best_epoch = 0
patience = 100  # 조기 중단을 결정할 epoch 횟수
best_timing_loss = 0
for epoch in range(num_epochs):
    # NumPy 배열로 변환
    inputs = torch.tensor(df_train).float()
    labels = torch.tensor(label_train.values).long()  # NumPy 배열로 변환 후 Tensor로 변환

    # 순전파
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 역전파 및 가중치 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    with torch.no_grad():
        test_inputs = torch.tensor(df_test).float()
        test_labels = torch.tensor(label_test.values).long()
        test_outputs = model(test_inputs)
        softmax = nn.Softmax(dim=1)
        softmax_outputs = softmax(test_outputs)
        _, predicted = torch.max(softmax_outputs, 1)
        total = test_labels.size(0)
        correct = (predicted == test_labels).sum().item()
        accuracy = correct / total
    if (epoch + 1) % 10 == 0:
        print('Test Accuracy: {:.4f}'.format(accuracy))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epoch + 1
        best_timing_loss = loss.item()
    elif epoch - best_epoch >= patience:
        print('Early Stopping at Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        print(best_accuracy)
        break
