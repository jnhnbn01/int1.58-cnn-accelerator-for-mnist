import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# --- GPU 디바이스 설정 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Custom CNN Model Using PyTorch Functional Layers ---
class IntQuantizedCNN(nn.Module):
    def __init__(self, conv1_weight, conv2_weight, fc_weight, conv1_q, conv2_q, fc_q, input_q):
        super(IntQuantizedCNN, self).__init__()
        self.input_q = input_q
        self.conv1_q = conv1_q
        self.conv2_q = conv2_q
        self.fc_q = fc_q
        # Prepare conv weights: (out_channels, in_channels, kernel_h, kernel_w)
        self.conv1_weight = nn.Parameter(torch.tensor(conv1_weight, dtype=torch.float32).unsqueeze(1), requires_grad=False)
        self.conv2_weight = nn.Parameter(torch.tensor(conv2_weight, dtype=torch.float32), requires_grad=False)
        self.fc_weight = nn.Parameter(torch.tensor(fc_weight, dtype=torch.int32), requires_grad=False)

    def forward(self, x):
        x = x.float()

        # Conv1: (B, 1, 28, 28) → (B, 8, 26, 26)
        x = F.conv2d(x, self.conv1_weight, bias=None, stride=1, padding=0)
        x = (x.to(torch.int32) >> self.conv1_q).clamp(-2**(7 - self.input_q), 2**(7 - self.input_q) - 1)
        x = F.relu(x)
        x = (x.to(torch.int32) << self.input_q)

        # Conv2: (B, 8, 26, 26) → (B, 16, 24, 24)
        x = F.conv2d(x.float(), self.conv2_weight, bias=None, stride=1, padding=0)
        x = (x.to(torch.int32) >> self.conv2_q).clamp(-2**(7 - self.input_q), 2**(7 - self.input_q) - 1)
        x = F.relu(x)
        x = (x.to(torch.int32) << self.input_q)

        # MaxPool: (B, 16, 24, 24) → (B, 16, 12, 12)
        x = F.max_pool2d(x.float(), kernel_size=2, stride=2).to(torch.int32)

        # Flatten
        x = x.view(x.size(0), -1)

        # Flatten and FC: (B, 2304) → (B, 10)
        x = x.cpu().to(torch.float32)
        fc_w = self.fc_weight.cpu().to(torch.float32)
        x = ((x @ fc_w.T).to(torch.int32) >> self.fc_q).clamp(-2**(7 - self.input_q), 2**(7 - self.input_q) - 1)
        x = (x.to(torch.int32) << self.input_q)
        return x

# --- Weight COE 파서 ---
def parse_signed_decimal_coe(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        if 'memory_initialization' in line or 'radix' in line:
            continue
        if ';' in line:
            line = line.split(';')[0]
        values = line.strip().replace(',', ' ').split()
        for val in values:
            if val:
                data.append(int(val))
    return np.array(data)

# --- 데이터 및 가중치 로딩 ---
input_images = torch.tensor(np.load("input.npy"), dtype=torch.int32).to(device)  # (10000, 1, 28, 28)
labels_vec = torch.tensor(np.load("label.npy"), dtype=torch.float32).to(device)  # (10000, 10)
labels = labels_vec

weight_data = parse_signed_decimal_coe("layer1_2_weight.coe")
conv1_weight = weight_data[:72].reshape(8, 3, 3)
conv2_weight = weight_data[72:].reshape(16, 8, 3, 3)
fc_weight = np.load("fc1_weight.npy")  # shape: (10, 2304)

# --- Quantization 테스트 루프 ---
for input_q in range(1, 8):
#for input_q in range(1, 8):
    print("Quantization to", 8 - input_q, "bits")
    for conv1_q in range(15, -1, -1):
        for conv2_q in range(15, -1, -1):
            for fc_q in range(15, -1, -1):
                # 모델 생성 및 GPU로 이동
                model = IntQuantizedCNN(conv1_weight, conv2_weight, fc_weight, conv1_q, conv2_q, fc_q, input_q).to(device)
                model.eval()

                with torch.no_grad():
                    inputs = (input_images >> input_q)
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    accuracy = (preds == labels.to(preds.device)).float().mean().item() * 100
                    print(conv1_q, "for conv1,", conv2_q, "for conv2,", fc_q, "for fc ->", f"Accuracy: {accuracy:.2f}%")
