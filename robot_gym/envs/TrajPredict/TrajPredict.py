import copy
import math
import torch
import pandas as pd
import numpy as np
import onnxruntime as ort
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------- 预测模型 ----------------------------------------
# 基于mlp作轨迹预测
class MLPTrajPredict(nn.Module):
    def __init__(self, output_dim=7, frame=10):
        super().__init__()
        self.input_dim = 7*frame
        self.hidden_dim = 256  # 你可以调整这个超参数
        self.output_dim = output_dim
        self.num_layers = 3
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),  # 第一隐藏层
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim//2),  # 第二隐藏层（维度递减设计）
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, self.hidden_dim//4),   # 输出层
            nn.ReLU(),
            nn.Linear(self.hidden_dim//4, self.output_dim)   # 输出层
        )
        
    def forward(self, x):
        x = self.mlp(x)
        return x

# 基于CNN作轨迹预测
class CNNTrajPredict(nn.Module):
    def __init__(self, output_dim=7):
        super(CNNTrajPredict, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 1 * 2, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 1 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 基于LSTM作轨迹预测
class LSTMTrajPredict(nn.Module):
    def __init__(self, output_dim=7):
        super(LSTMTrajPredict, self).__init__()
        self.input_dim = 7
        self.hidden_dim = 64  # 你可以调整这个超参数
        self.output_dim = output_dim
        self.num_layers = 2
        self.dropout_rate=0.2
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, self.output_dim)
        )

        # self.lstm1 = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True)
        # self.dropout1 = nn.Dropout(self.dropout_rate)
        
        # # 第二层LSTM + Dropout
        # self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        # self.dropout2 = nn.Dropout(self.dropout_rate)
        
        # # 输出层
        # self.fc = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x):
        # LSTM过程 -> batch_first表示输入的格式为 (batch_size, seq_length, input_dim)
        # lstm_out, _ = self.lstm(x)  # 我们不需要hidden state
        # 从LSTM的输出中取出最后一个时间步的隐藏状态
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        # 使用全连接层进行进一步处理
        output = self.fc(last_hidden_state)

        # x, _ = self.lstm1(x)
        # x = self.dropout1(x)  # 层间Dropout
        # x, _ = self.lstm2(x)
        # x = self.dropout2(x)  # 层间Dropout
        # x = self.fc(x[:, -1, :])  # 取序列最后一步输出
        return output

# 基于GRU作轨迹预测
class GRUTrajPredict(nn.Module):
    def __init__(self, output_dim=7):
        super(GRUTrajPredict, self).__init__()
        self.input_dim = 7
        self.hidden_dim = 64  # 你可以调整这个超参数
        self.output_dim = output_dim
        self.num_layers = 2
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, self.output_dim)
        )

    def forward(self, x):
        # x 形状： (batch_size, sequence_length, input_size)
        out, _ = self.gru(x)
        # out 形状： (batch_size, sequence_length, hidden_size)
        out = out[:, -1, :]  # 获取最后一个隐藏状态
        # out 形状： (batch_size, hidden_size)
        out = self.fc(out)
        # out 形状： (batch_size, output_size)
        return out



# ------------------------------------------- 数据处理 ----------------------------------------
class DataLoad:
    def __init__(self, datafile):
        self.lens = 1000
        self.Data, self.Labels = self.loadfromfile(datafile)
        self.TrainData = self.Data[0:self.lens]
        self.TrainLabel = self.Labels[0:self.lens]
        self.TestData = self.Data[self.lens:]
        self.TestLabel = self.Labels[self.lens:]
        self.Stard_flag = False
        self.Normal_flag = False

        print(self.TrainData[0][0])
        print(self.TrainLabel[0][0])

        if self.Stard_flag:
            print("======================= Standardization ===================")
            self.train_x_normal_coef, self.train_label_normal_coef, self.train_x_normal, self.train_label_normal = \
                self.Standardization(self.TrainData.copy(), self.TrainLabel.copy(), "train")
            
            self.test_x_normal_coef, self.test_label_normal_coef, self.test_x_normal, self.test_label_normal = \
                self.Standardization(self.TestData.copy(), self.TestLabel.copy(), "test")
            
            print("normal: ", self.train_x_normal.shape, self.train_label_normal.shape)
            print("normal data mean: ", self.train_x_normal_coef.mean_, self.train_x_normal_coef.scale_)
            print("normal label mean: ", self.train_label_normal_coef.mean_, self.train_label_normal_coef.scale_)
            # print("normal test: ", self.test_normal.shape, self.test_label_normal.shape)
            # print("normal test mean: ", self.normal_test.mean_, self.normal_test.scale_)

            self.TrainData = self.train_x_normal.copy()
            self.TrainLabel = self.train_label_normal.copy()
            self.TestData = self.test_x_normal.copy()
            self.TestLabel = self.test_label_normal.copy()

            print(self.TrainData[0][0])
            print(self.TrainLabel[0][0])
        if self.Normal_flag:
            print("======================= Normalization ===================")
            train_data_normal, train_label_normal = self.Normalization(self.TrainData.copy(), self.TrainLabel.copy())
            test_data_normal, test_label_normal = self.Normalization(self.TestData.copy(), self.TestLabel.copy())
            
            self.TrainData = train_data_normal.copy()
            self.TrainLabel = train_label_normal.copy()
            self.TestData = test_data_normal.copy()
            self.TestLabel = test_label_normal.copy()
            print(self.TrainData[0][0])
            print(self.TrainLabel[0][0])

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def loadfromfile(self, datafile):
        data = pd.read_parquet(datafile)

        train_data = data['TrainData']
        label = data['DataLabel']
        TrainData = []
        DataLabel = []
        for i in range(len(train_data)):
            if len(label[i])!=0:
                traj = train_data[i]
                traj = np.vstack(traj)
                traj = traj.tolist()

                labeltmp = label[i]
                labeltmp = np.vstack(labeltmp)
                labeltmp = labeltmp.tolist()

                TrainData.append(traj)
                DataLabel.append(labeltmp)
            # else:
            #     print(i)

        TrainData = np.array(TrainData)
        DataLabel = np.array(DataLabel)
        
        return TrainData, DataLabel
    
    def Normalization(self, Data, Label):
        min = np.array([-0.23, 3.9, 0.6, -1.0, -11.0, 5.0, 0.0])
        max = np.array([-0.03, 4.1, 0.8, 1.0, -6.0, 9.0, 1.0])
        Data_Normal = Data.copy()
        Label_Normal = Label.copy()
        data_len = Data.shape
        label_len = Label.shape
        print(Data_Normal[1][0])
        print(Label_Normal[1])
        for i in range(data_len[0]):
            for j in range(data_len[1]):
                Data_Normal[i][j] = (Data[i][j] - min)/(max - min)

        for i in range(label_len[0]):
            Label_Normal[i] = (Label[i] - min)/(max - min)
        
        return Data_Normal, Label_Normal

    def Standardization(self, Data, Label, flag):
        if flag == "train":
            scaler_X = StandardScaler()
            scaler_label = StandardScaler()
            # 仅在训练集上拟合
            X_train_scaled = scaler_X.fit_transform(Data.reshape(-1,Data.shape[-1])).reshape(Data.shape)  # 输入特征标准化
            Label_train_scaled = scaler_label.fit_transform(Label.reshape(-1,Data.shape[-1])).reshape(Label.shape)  # 目标值标准化
        elif flag=="test":
            scaler_X = StandardScaler()
            scaler_label = StandardScaler()
            scaler_X.mean_ = self.train_x_normal_coef.mean_    # 直接赋值均值
            scaler_X.scale_ = self.train_x_normal_coef.scale_    # 直接赋值标准差
            scaler_label.mean_ = self.train_label_normal_coef.mean_    # 直接赋值均值
            scaler_label.scale_ = self.train_label_normal_coef.scale_    # 直接赋值标准差

            # 测试集拟合
            X_train_scaled = scaler_X.transform(Data.reshape(-1,Data.shape[-1])).reshape(Data.shape)  # 输入特征标准化
            Label_train_scaled = scaler_label.transform(Label.reshape(-1,Data.shape[-1])).reshape(Label.shape)  # 目标值标准化
        return scaler_X, scaler_label, X_train_scaled, Label_train_scaled


class CustomDataset(Dataset):
    def __init__(self, data, labels, Policy):
        if Policy == "lstm" or Policy == "gru":
            self.data = torch.tensor(data, dtype=torch.float32)  # (数量, 时间长度, 状态维度)
            self.labels = torch.tensor(labels, dtype=torch.float32).squeeze(1)  # (数量, 状态维度)
        elif Policy == "cnn":
            # 转换数据为张量并调整形状到 (10000, 1, 7, 15)
            self.data = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1).unsqueeze(1)
            # 转换标签为张量并去掉多余维度到 (10000, 7)
            self.labels = torch.tensor(labels, dtype=torch.float32).squeeze(1)
        elif Policy == "mlp":
            print(data.shape, labels.shape)
            # 转换数据为张量并调整形状到 (10000, 70)
            # self.data = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1).unsqueeze(1)
            data_flatten = data.reshape(data.shape[0], -1)
            self.data = torch.tensor(data_flatten, dtype=torch.float32)
            # 转换标签为张量并去掉多余维度到 (10000, 7)
            self.labels = torch.tensor(labels, dtype=torch.float32).squeeze(1)
            pass
        print(self.data.shape, self.labels.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# -------------------------------------------- 训练 ------------------------------------------
# 学习率预热 + 衰减
def lr_schedule(epoch,optimizer):
    warmup_epochs = 5
    decay_rate = 0.95
    initial_lr=0.001
    if epoch < warmup_epochs:
        lr = initial_lr * (epoch + 1) / warmup_epochs  # 线性升温
    else:
        lr = initial_lr * (decay_rate**(epoch - warmup_epochs))  # 指数衰减
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # print("lr: ", lr)
    return lr

def train(Policy):
    # data load
    filepath = '/home/yyy/Documents/BadmintonMatch'
    # datafile = filepath + '/data/BadmintonMatch/TrajPredict/TrainData_0_02_large.parquet'
    # datafile = filepath + '/data/BadmintonMatch/TrajPredict/0227/TrainAndLabel_0227.parquet'
    # datafile = filepath + '/data/BadmintonMatch/TrajPredict/0227/TrainAndLabel_0227_4D.parquet'
    # datafile = filepath + '/data/BadmintonMatch/TrajPredict/0421/TrainAndLabel_0421_4D_10.parquet'
    frame = 10
    datafile = filepath + '/data/BadmintonMatch/TrajPredict/0515/TrainAndLabel_0515_4D_'+ str(frame) + '.parquet'
    Data = DataLoad(datafile)
    print(Data.TrainData.shape, Data.TrainLabel.shape)
    dataset = CustomDataset(Data.TrainData, Data.TrainLabel, Policy)
    testdataset = CustomDataset(Data.TestData, Data.TestLabel, Policy)

    # a=df
    # Define hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 200
    output_dim = 4

    if Policy == "lstm":
        print("======================= LSTM Policy ===================")
        model = LSTMTrajPredict(output_dim)
    elif Policy == "cnn":
        print("======================= CNN Policy ===================")
        model = CNNTrajPredict(output_dim)
    elif Policy == "gru":
        print("======================= GRU Policy ===================")
        model = GRUTrajPredict(output_dim)
    elif Policy == "mlp":
        print("======================= MLP Policy ===================")
        model = MLPTrajPredict(output_dim, frame)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 余弦退火调整学习率
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    # Training loop
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # 线性预热 + 指数衰减
            lr_schedule(epoch, optimizer)
            
            # 余弦退火调整学习率
            # scheduler.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # Evaluate the model
    model.eval()   
    total_loss = 0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)  # 前向传播
            loss = criterion(outputs, batch_labels)  # 计算损失
            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)
    print(f'平均测试损失: {average_loss:.4f}')

    torch.save(model.state_dict(), filepath+'/config/0515/'+Policy+'_4D_' + str(frame) +'.pth')

    print(Policy)

# -------------------------------------------- onnx模型导出 ------------------------------------------
# 导出模型
def onnximport(Policy):
    filepath = '/home/yyy/Documents/BadmintonMatch'
    output_dim = 4
    frame = 10
    # 载入参数
    if Policy == "lstm":
        model = LSTMTrajPredict(output_dim)
        new_data = np.random.randn(1, 10, 7)
    elif Policy == "cnn":
        model = CNNTrajPredict(output_dim)
        new_data = np.random.randn(1, 1, 7, 10)
    elif Policy == "gru":
        model = GRUTrajPredict(output_dim)
        new_data = np.random.randn(1, 10, 7)
    elif Policy == "mlp":
        model = MLPTrajPredict(output_dim, frame)
        new_data = np.random.randn(1, 7*frame)
        
    model.load_state_dict(torch.load(filepath+'/config/0515/'+Policy+'_4D_' + str(frame) +'.pth', weights_only=True))
    model.eval()  # 切换到评估模式

    # 准备输入数据 (假设新数据保持 `(seq_length, input_dim)` 格式)
    # 示例数据：一个批次大小为1的样例 (1, seq_length, input_dim)
    # new_data = np.random.randn(1, 10, 7)  # 其中15是 seq_length 可以根据需要调整
    input = torch.tensor(new_data, dtype=torch.float32)

    # 导出模型为 ONNX 格式
    onnx_file_path =filepath+'/config/0515/'+Policy+'_4D_' + str(frame) +'.onnx'
    torch.onnx.export(
        model,                 # 要转换的模型
        input,           # 示例输入张量
        onnx_file_path,        # 导出文件的路径
        input_names=['input'], # 输入节点的名称
        output_names=['output'], # 输出节点的名称
        opset_version=13,       # 使用的 ONNX opset 版本
        dynamic_axes=None
    )




if __name__=='__main__':
    Policy = "mlp"
    train(Policy)
    # onnximport(Policy)


    
"""
5:  mean:  [0.083, 0.0048, 0.083, 0.034]
    std:  [0.118, 0.003, 0.076, 0.025]
10: mean:  [0.072, 0.00695, 0.081, 0.026]
    std:  [0.080, 0.0028, 0.074, 0.023]
"""
    