CNN Accelerator for MNIST Classification

Running inference of very simple CNN(conv3-conv3-maxpool-fc) on FPGA
INT8 means the baseline int8 quantized model
Ternary means the ternary quantized model that we modified the baseline model and applied QAT

- In the 'bitstreams' folder, there are bitstream files and you can test the accelerator upload it onto PYNQ-Z2 board.
- In the 'trained_weights' there are .npy files including weights of ternary quantized and baseline models.
- In the 'hdl_codes' folder, there are Verilog codes of the accelerator that we designed. Due to the large file size of our project, I uploaed ip folder in thse project, including entire verilog codes.
- In the 'quantization_codes' folder, you can find the PyTorch implementation of ternary QAT for the baseline model.


Please refer to Assignment4_announcement.pdf for the full project specification and to Report.pdf for our implementation details and results.
