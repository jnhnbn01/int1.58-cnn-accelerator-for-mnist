import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from collections import Counter


# === 캐시 비우기
torch.cuda.empty_cache() 

# === 경로 설정 ===
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(script_dir, "quan_data"))
device = 'cpu'
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MNIST 데이터셋 로드 ===
mnist_dataset = torchvision.datasets.MNIST(
    root='MNIST',
    train=True,
    download=False,  # 이미 다운받았다고 하셨으므로 False
    transform=transforms.ToTensor()
)

# === 전체 데이터 수 확인 ===
total_data = len(mnist_dataset)  # 일반적으로 60,000장
print(f"전체 MNIST 데이터 개수: {total_data}")

# === 랜덤 인덱스 10,000개 선택 ===
rand_indices = torch.randperm(total_data)[:10000]

# === 이미지/라벨 추출 ===
images = []
labels = []

for idx in rand_indices:
    image, label = mnist_dataset[idx]
    images.append(image)
    labels.append(label)

# === 텐서로 변환 ===
images_tensor = torch.stack(images)              # (10000, 1, 28, 28)
labels_tensor = torch.tensor(labels, dtype=torch.long)  # (10000,)

print(f"이미지 텐서 shape: {images_tensor.shape}")
print(f"라벨 텐서 shape: {labels_tensor.shape}")

np.save("coe_files/input_random.npy", images_tensor.numpy())   # shape: (10000, 1, 28, 28)
np.save("coe_files/label_random.npy", labels_tensor.numpy())   # shape: (10000,)


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
        self.act1 = TernaryActivation(init_alpha=0.8, init_threshold=0.25)
        self.conv2 = TernaryConv2d(8, 16, 3, init_alpha=0.5, init_threshold=0.0625)
        self.relu2 = nn.LeakyReLU(0.5)
        self.act2 = TernaryActivation(init_alpha=1, init_threshold=0.25)
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

# === coe 파일 로드 함수 ===
def load_coe(filename, shape):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data_lines = lines[2:]  # 첫 두 줄은 header
    raw = ",".join([l.strip().rstrip(';') for l in data_lines])
    values = np.array([int(x) for x in raw.split(",") if x.strip() != ""])
    return values.reshape(shape)

# === ternary dequantization 함수 ===
def apply_ternary(x, alpha):
    return x * alpha  # x ∈ {-1, 0, +1}

# === 모델 로드 및 alpha 추출 ===
print("[1] loading trained model...")
model = TernaryCNN().to(device)
model.eval()
model.load_state_dict(torch.load("ternary_qat_cnn.pth", map_location=device))


# alpha 값 출력
alpha_input = model.input_quant.alpha.item()
alpha_conv1 = model.conv1.alpha.item()
alpha_conv2 = model.conv2.alpha.item()
alpha_fc = 1
alpha_activation1 = model.act1.alpha.item()
alpha_activation2 = 1

# threshold 값 출력
threshold_input = model.input_quant.threshold.item()
threshold_conv1 = model.conv1.threshold.item()
threshold_conv2 = model.conv2.threshold.item()
threshold_fc = model.fc.threshold.item()
threshold_activation1 = model.act1.threshold.item()
threshold_activation2 = model.act2.threshold.item()

# alpha와 threshold 출력
print("\n[Alpha Values]")
print(f"Alpha (Input Activation): {alpha_input}")
print(f"Alpha (Conv1): {alpha_conv1}")
print(f"Alpha (Conv2): {alpha_conv2}")
print(f"Alpha (FC): {alpha_fc}")
print(f"Alpha (Activation1): {alpha_activation1}")
print(f"Alpha (Activation2): {alpha_activation2}")

print("\n[Threshold Values]")
print(f"Threshold (Input Activation): {threshold_input}")
print(f"Threshold (Conv1): {threshold_conv1}")
print(f"Threshold (Conv2): {threshold_conv2}")
print(f"Threshold (FC): {threshold_fc}")
print(f"Threshold (Activation1): {threshold_activation1}")
print(f"Threshold (Activation2): {threshold_activation2}")

# === coe 로드 ===
print("[2] loading .coe files...")
input = np.where(np.load("coe_files/input_random.npy") == 0, 0, 1)
conv1_weight = load_coe("coe_files/conv1_weight.coe", (8, 1, 3, 3))
conv2_weight = load_coe("coe_files/conv2_weight.coe", (16, 8, 3, 3))
fc_weight = load_coe("coe_files/fc_weight.coe", (10, 2304))

# === 분포 확인 ===
print("[3] checking quantized values...")
print("   Input:", Counter(input.flatten()))
print("   Conv1:", Counter(conv1_weight.flatten()))
print("   Conv2:", Counter(conv2_weight.flatten()))
print("   FC:", Counter(fc_weight.flatten()))

# === 텐서 변환 (dequantization) ===
input_tensor = torch.tensor(apply_ternary(input, alpha_input), dtype=torch.float32).to(device)
conv1_w = torch.tensor(apply_ternary(conv1_weight, alpha_conv1), dtype=torch.float32).to(device)
conv2_w = torch.tensor(apply_ternary(conv2_weight, alpha_conv2), dtype=torch.float32).to(device)
fc_w = torch.tensor(apply_ternary(fc_weight, 1), dtype=torch.float32).to(device)

# === 정답 로드 ===
true_labels = torch.tensor(np.load("coe_files/label_random.npy"), dtype=torch.long).to(device)
#true_labels = torch.argmax(output_tensor, dim=1)

# === 추론 ===
print("[4] running inference with dequantized weights and learned alpha...")


with torch.no_grad():
    # 첫 번째 convolutional layer
    x = F.conv2d(input_tensor, conv1_w, stride=1, padding=0)
    print("   After conv1:", x.shape)
    unique_values, counts = torch.unique(x, return_counts=True)

    # 첫 번째 activation 통과
    x = model.relu1(x)
    x = model.act1(x)
    print("   After act1:", x.shape)
    unique_values, counts = torch.unique(x, return_counts=True)
    dout_conv1 = (x / 0.8).round().to(torch.int32)

    # 두 번째 convolutional layer
    x = F.conv2d(x, conv2_w, stride=1, padding=0)
    print("   After conv2:", x.shape)
    unique_values, counts = torch.unique(x, return_counts=True)

    # 두 번째 activation 통과
    x = model.relu2(x)
    x = model.act2(x) 
    print("   After act2:", x.shape)
    dout_conv2 = (x / 1.25).round().to(torch.int32)

    # Pooling 후
    x = model.pool(x)
    print("   After pool:", x.shape)
    dout_pool = (x / 1.25).round().to(torch.int32)

    # 최종 분류 결과
    x = x/1.25
    x = x.view(x.size(0), -1)
    logits = F.linear(x, fc_w)
    logits = logits.to(torch.int16) 
    logits = torch.clamp(logits >> 1, min=-128, max=127)
    predicted = torch.argmax(logits, dim=1)
    accuracy = (predicted == true_labels).float().mean().item() * 100
    print(f"Accuracy: {accuracy:.2f}%")
# === 결과 출력 ===
print("\n[Inference 결과]")
#print(f"정확도: {accuracy:.2f}%")

def save_coe_file(data, filename, shape):
    os.makedirs("coe_files", exist_ok=True)
    data = data.cpu().detach().numpy().reshape(shape).flatten()
    data = data.astype(int)  # 1차원으로 펼침
    with open(filename, 'w') as f:
        f.write("memory_initialization_radix=10;\n")
        f.write("memory_initialization_vector=\n")
        f.write(",\n".join(map(str, data)) + ";\n")
        
save_coe_file(logits,"coe_files/output_logits.coe", logits.shape)
print(f"logits : {logits.cpu().detach().numpy()}")
print("\n[Logits 결과]")
print(f"Min value of logits: {logits.min().item()}")
print(f"Max value of logits: {logits.max().item()}")
print(f"labels : {true_labels}")

save_coe_file(dout_conv1,"coe_files/dout_conv1.coe", dout_conv1.shape)
save_coe_file(dout_conv2,"coe_files/dout_conv2.coe", dout_conv2.shape)
save_coe_file(dout_pool,"coe_files/dout_pool.coe", dout_pool.shape)