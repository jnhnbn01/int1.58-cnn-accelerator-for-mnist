import numpy as np

# 파일 경로 지정
input_data = np.load('input.npy')
conv1_weight = np.load('layer1_0_weight.npy')
conv2_weight = np.load('layer2_0_weight.npy')
affine_weight = np.load('fc1_weight.npy')
output_data = np.load('output.npy')



# 데이터 크기출력
print("Input Data:", input_data.shape)
print("Output Data:", output_data.shape)
print("Convolution Layer 1 Weight:", conv1_weight.shape)
print("Convolution Layer 2 Weight:", conv2_weight.shape)
print("Affine Layer 1 Weight:", affine_weight.shape)

# 원하는 데이터의 index 지정
input_data_index = 0
output_data_index = 0
conv2_weight_ichannel = 0
conv2_weight_ochannel = 1
     
input_fmap = np.load("input.npy")
conv1_weight = np.load("layer1_0_weight.npy")
conv2_weight = np.load("layer2_0_weight.npy")
fc_weight = np.load("fc1_weight.npy")

conv12_weight = np.concatenate((conv1_weight.flatten(), conv2_weight.flatten()))
fc_weight1 = fc_weight[0].flatten()
fc_weight2 = fc_weight[1, :].flatten()
fc_weight3 = fc_weight[2, :].flatten()
fc_weight4 = fc_weight[3, :].flatten()
fc_weight5 = fc_weight[4, :].flatten()
fc_weight6 = fc_weight[5, :].flatten()
fc_weight7 = fc_weight[6, :].flatten()
fc_weight8 = fc_weight[7, :].flatten()
fc_weight9 = fc_weight[8, :].flatten()
fc_weight10 = fc_weight[9, :].flatten()

print(output_data[0:100])
