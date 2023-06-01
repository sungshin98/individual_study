import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim

#hello
def pre_df(df):
    df = df.dropna()
    label = df['Àç½ÇÀÎ¿ø']
    df = df.drop('Àç½ÇÀÎ¿ø', axis=1)
    df = df.drop('regdate', axis=1)
    df = df.drop('PIR', axis=1)
    df_33 = df.values.reshape(-1,3,3)
    return df_33, label
#0~255 스케일링 함수
def scale_255(array,i,j):
    # i,j 위치의 요소들을 추출하여 배열 생성
    first_elements = array[:, i, j]
    # 첫 번째 요소들의 최솟값과 최댓값 계산
    min_value = np.min(first_elements)
    max_value = np.max(first_elements)
    # 스케일링 수행
    scaled_array = (first_elements - min_value) * (255 / (max_value - min_value))
    return scaled_array
def make_28(df_33):
    # 결과 행렬을 저장할 빈 배열 생성
    expanded_data = np.zeros((len(df_33), 28, 28))
    # df_33의 각 3x3 배열에 대해 반복
    for idx, arr in enumerate(df_33):
        expanded_arr = np.repeat(arr,9,axis=0)
        expanded_arr = np.repeat(expanded_arr,9,axis=1)
        # 패딩을 추가한 새로운 28x28 배열 생성
        expanded_arr = np.pad(expanded_arr, ((0, 1), (0, 1)), mode='constant')
        expanded_data[idx] = expanded_arr
    return expanded_data

df = pd.read_csv("train_data.csv", encoding='ISO-8859-1')
df_33, label = pre_df(df)
#스케일링 함수 전체 적용
for i in range(3):
    for j in range(3):
        df_33[:,i,j] = scale_255(df_33,i,j)
df_33 = make_28(df_33)
# 배열을 PyTorch Tensor로 변환
input_data = torch.from_numpy(df_33).unsqueeze(1).float()  # 차원 수정
# 라벨 데이터를 PyTorch Tensor로 변환
target = torch.tensor(label.values)
print(input_data.shape)
#test
df_test = pd.read_csv("test_data.csv", encoding='ISO-8859-1')
df_test_33, label_test = pre_df(df_test)
#스케일링 함수 전체 적용
for i in range(3):
    for j in range(3):
        df_test_33[:,i,j] = scale_255(df_test_33,i,j)
df_test_33 = make_28(df_test_33)
# 배열을 PyTorch Tensor로 변환
input_test = torch.from_numpy(df_test_33).unsqueeze(1).float()  # 차원 수정
# 라벨 데이터를 PyTorch Tensor로 변환
target_test = torch.tensor(label_test.values)
print(input_test.shape)

# LeNet-5 모델 정의 (시퀀셜)
model = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 400),
    nn.ReLU(),
    nn.Linear(400, 120),
    nn.ReLU(),
    nn.Linear(120, 7)
)

# 모델 학습 또는 예측 수행

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 학습
num_epochs = 10
for epoch in range(num_epochs):
    # 입력 데이터의 forward pass
    outputs = model(input_data)

    # 손실 계산
    loss = criterion(outputs, target)

    # 역전파 및 가중치 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 예측
    with torch.no_grad():
        outputs = model(input_test)
        _, predicted = torch.max(outputs.data, 1)

    # 예측 결과와 실제 라벨 비교하여 정확도 계산
    accuracy = (predicted == target_test).sum().item() / target_test.size(0)

    # 현재 에폭의 손실과 정확도 출력
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Accuracy: {accuracy}')

