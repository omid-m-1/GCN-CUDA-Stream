# 2-Layer GCN with CUDA Kernel Running on the Same Stream as PyTorch Uses - Assignment 5

This repository contains the implementation of a 2-layer Graph Convolutional Network (GCN) using a CUDA kernel. In this assignment, PyTorch and CUDA operations are synchronized to run on the same stream for better performance and efficiency.

## Usage

To train the GCN model, run the following command:

```bash
python main.py --dataset <dataset>
```

Replace `<dataset>` with the name of the desired dataset. Supported datasets include:
- `cora`
- `citeseer`
- `pubmed`
- `reddit`

The results for each dataset are saved in the `result_stream` folder with the naming convention `<dataset>_stream.txt`.

## Compiling the Kernel

To compile the CUDA kernel, run the following commands from the `deep-codegen` directory:

```bash
mkdir build && cd build
cmake ..
make -j
cp graphpy.cpython-38-x86_64-linux-gnu.so ../
```

This will build the required shared library (`graphpy.cpython-38-x86_64-linux-gnu.so`) for the Python bindings.

## Ensuring PyTorch and CUDA Run on the Same Stream

This implementation ensures that PyTorch and CUDA operations run on the same stream using PyTorch’s `torch.cuda.current_stream()` API. The current stream is passed from PyTorch to the CUDA backend during kernel launches.

## Result Directory `result_stream`
Each file in result_stream directory contains:
- The memory address of the CUDA stream used during computation, verifying that PyTorch and CUDA operations ran on the same stream.
- Epoch-wise training loss and accuracy.
- Final test loss and accuracy.

### Example (Cora Dataset)
Below is a snippet from `cora_stream.txt`:

```
CUDA Stream ID: 0x7d24c00
PyTorch Stream ID: 0x7d24c00
Epoch [5000] - Train Loss: 0.0060625 | Train Accuracy: 84.29%
Epoch [10000] - Train Loss: 0.0022841 | Train Accuracy: 92.14%
Test Loss: 0.00068 | Test Accuracy: 79.50%
```

This output confirms that PyTorch and CUDA are using the same stream (`0x7d24c00`) and provides key training and test results for the `cora` dataset.
