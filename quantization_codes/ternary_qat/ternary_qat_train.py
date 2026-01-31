import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 작업 디렉토리 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(script_dir, "quan_data"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, threshold):
        ctx.save_for_backward(input, alpha, threshold)

        out = torch.zeros_like(input)
        out[input > threshold] = alpha
        out[input < -threshold] = -alpha
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha, threshold = ctx.saved_tensors
        grad_input = grad_output.clone()

        # === grad_input ===
        grad_input[torch.abs(input) > alpha] = 0

        # === grad_alpha ===
        grad_alpha = grad_output.clone()
        grad_alpha_mask = ((input > threshold) | (input < -threshold)).float()
        grad_alpha = torch.sum(grad_alpha * grad_alpha_mask)

        # === grad_threshold ===
        grad_threshold = grad_output.clone()
        grad_threshold_mask = ((torch.abs(input - threshold) < 1e-3) | (torch.abs(input + threshold) < 1e-3)).float()
        grad_threshold = -torch.sum(grad_threshold * grad_threshold_mask)

        return grad_input, grad_alpha, grad_threshold

# === TernaryConv2d ===
class TernaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, init_alpha=1.0, init_threshold=0.1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.threshold = nn.Parameter(torch.tensor(init_threshold))

    def forward(self, x):
        w_q = self.weight + (STEFunction.apply(self.weight, self.alpha, self.threshold) - self.weight).detach()
        return F.conv2d(x, w_q, stride=self.stride, padding=self.padding)

# === TernaryLinear ===
class TernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, init_alpha=1.0, init_threshold=0.1):
        super().__init__(in_features, out_features, bias=False)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.threshold = nn.Parameter(torch.tensor(init_threshold))

    def forward(self, x):
        w_q = self.weight + (STEFunction.apply(self.weight, self.alpha, self.threshold) - self.weight).detach()
        return F.linear(x, w_q)

# === TernaryActivation ===
class TernaryActivation(nn.Module):
    def __init__(self, init_alpha=1.0, init_threshold=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float32))


    def forward(self, x):
        out = torch.zeros_like(x)
        out[x > self.threshold] = self.alpha
        out[x < -self.threshold] = -self.alpha
        return x + (out - x).detach()

# === CNN 모델 ===
class TernaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_quant = TernaryActivation(init_alpha=1.0, init_threshold=0)
        self.conv1 = TernaryConv2d(1, 8, 3, init_alpha=0.5, init_threshold=0.09)
        self.relu1 = nn.LeakyReLU(0.5)
        self.act1 = TernaryActivation(init_alpha=0.8, init_threshold=0.125)
        self.conv2 = TernaryConv2d(8, 16, 3, init_alpha=0.5, init_threshold=0.0625)
        self.relu2 = nn.LeakyReLU(0.5)
        self.act2 = TernaryActivation(init_alpha=1.0, init_threshold=0.25)
        self.pool = nn.MaxPool2d(2)
        self.fc = TernaryLinear(16*12*12, 10, init_alpha=0.75, init_threshold=0.025)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.act2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# === COE 저장 함수 ===
def save_coe_file(data, filename, shape):
    os.makedirs("coe_files", exist_ok=True)
    data = data.cpu().detach().numpy().reshape(shape)
    data = np.sign(data).astype(int).flatten()
    with open(filename, 'w') as f:
        f.write("memory_initialization_radix=10;\n")
        f.write("memory_initialization_vector=\n")
        f.write(",\n".join(map(str, data)) + ";\n")

# === ternary화 후 coe 저장 함수 (alpha, threshold 기반) ===
def quantize_and_save_weight(weight_tensor, alpha, threshold, filename, shape):
    os.makedirs("coe_files", exist_ok=True)
    weight_np = weight_tensor.detach().cpu().numpy().reshape(shape)
    
    # ternary quantization
    ternary = np.zeros_like(weight_np)
    ternary[weight_np > threshold] = 1
    ternary[weight_np < -threshold] = -1

    # coe 저장 (기호 그대로, int형)
    flat = ternary.astype(int).flatten()
    with open(filename, 'w') as f:
        f.write("memory_initialization_radix=10;\n")
        f.write("memory_initialization_vector=\n")
        f.write(",\n".join(map(str, flat)) + ";\n")

# === 데이터 로드 ===
input_data = np.load("input.npy")
output_data = np.load("label.npy")
input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
output_tensor = torch.tensor(output_data, dtype=torch.long).to(device)

# === 모델 정의 ===
model = TernaryCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === 학습 ===
num_epochs = 50
batch_size = 50
for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total_samples = 0

    # === alpha, threshold 이전 값 저장 ===
    prev_alpha = model.conv1.alpha.item()
    prev_th = model.conv1.threshold.item()

    for i in range(0, len(input_tensor), batch_size):
        inputs = input_tensor[i:i+batch_size]
        labels = output_tensor[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # === 디버깅용 출력: gradient 확인 ===
        #print(f"[DEBUG] Grad alpha (conv1): {model.conv1.alpha.grad}")
        #print(f"[DEBUG] Grad threshold (conv1): {model.conv1.threshold.grad}")

        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    acc = 100.0 * total_correct / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")
    # === 디버깅용 출력: 값이 실제로 바뀌는지 확인 ===
    #print(f"[DEBUG] Alpha conv1 change: {prev_alpha:.5f} -> {model.conv1.alpha.item():.5f}")
    #print(f"[DEBUG] Thresh conv1 change: {prev_th:.5f} -> {model.conv1.threshold.item():.5f}")
    #print("-" * 50)

# 모델 저장 후 ternary화하여 coe 저장
quantize_and_save_weight(model.conv1.weight, model.conv1.alpha.item(), model.conv1.threshold.item(), "coe_files/conv1_weight.coe", (8,1,3,3))
quantize_and_save_weight(model.conv2.weight, model.conv2.alpha.item(), model.conv2.threshold.item(), "coe_files/conv2_weight.coe", (16,8,3,3))
quantize_and_save_weight(model.fc.weight, model.fc.alpha.item(), model.fc.threshold.item(), "coe_files/fc_weight.coe", (10, 2304))


# === 학습된 alpha, threshold 출력 ===
print("=== Learned alpha / threshold values ===")
print(f"Conv1: alpha = {model.conv1.alpha.item():.4f}, threshold = {model.conv1.threshold.item():.4f}")
print(f"Conv2: alpha = {model.conv2.alpha.item():.4f}, threshold = {model.conv2.threshold.item():.4f}")
print(f"FC   : alpha = {model.fc.alpha.item():.4f}, threshold = {model.fc.threshold.item():.4f}")
print(f"Act1 : alpha = {model.act1.alpha.item():.4f}, threshold = {model.act1.threshold.item():.4f}")
print(f"Act2 : alpha = {model.act2.alpha.item():.4f}, threshold = {model.act2.threshold.item():.4f}")
print(f"Conv1_weight shape : {model.conv1.weight.shape}, min: {model.conv1.weight.min().item():.4f}, max: {model.conv1.weight.max().item():.4f}")
print(f"Conv2_weight shape : {model.conv2.weight.shape}, min: {model.conv2.weight.min().item():.4f}, max: {model.conv2.weight.max().item():.4f}")
print(f"FC_weight    shape : {model.fc.weight.shape}, min: {model.fc.weight.min().item():.4f}, max: {model.fc.weight.max().item():.4f}")

# === 학습 끝난 후에만 input 양자화 적용하여 저장 ===
input_quant = STEFunction.apply(input_tensor, torch.tensor(1.0).to(device), torch.tensor(0.1).to(device))
save_coe_file(input_quant, "coe_files/input_quant.coe", (10000, 1, 28, 28))

# === 모델 저장 ===
torch.save(model.state_dict(), "ternary_qat_cnn.pth")
