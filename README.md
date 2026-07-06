# 🧠 CNN From Scratch in C++

A **Convolutional Neural Network** built entirely from scratch in C++ without any deep learning framework. Trained and tested on the **MNIST** handwritten digit dataset.

The goal of this project is to understand the inner workings of CNNs by implementing every layer — convolution, pooling, and fully connected — along with their forward and backward passes manually.

---

## 📐 Network Architecture

```
Input (28×28) → Conv2D (3×3, ReLU) → MaxPool (2×2) → Flatten → Dense (169→64, Sigmoid) → Dense (64→10, Sigmoid) → Output
```

| Layer | Input | Output | Parameters | Activation |
|-------|-------|--------|------------|------------|
| Convolution | 28×28 | 26×26 | 10 | ReLU |
| Max Pooling | 26×26 | 13×13 | 0 | — |
| Flatten | 13×13 | 169 | 0 | — |
| Hidden (FC) | 169 | 64 | 10,880 | Sigmoid |
| Output (FC) | 64 | 10 | 650 | Sigmoid |
| **Total** | | | **11,540** | |

---

## 📊 Training Results

Trained on **3,000** images for **50 epochs** with a learning rate of **0.005**.

<details>
<summary>Training Log (click to expand)</summary>

```
Epoch: 0        | Average Error: 1.12609
Epoch: 1        | Average Error: 0.812946
Epoch: 2        | Average Error: 0.758806
Epoch: 3        | Average Error: 0.705263
Epoch: 4        | Average Error: 0.654006
Epoch: 5        | Average Error: 0.607336
Epoch: 10       | Average Error: 0.464426
Epoch: 15       | Average Error: 0.380022
Epoch: 20       | Average Error: 0.328199
Epoch: 25       | Average Error: 0.292130
Epoch: 30       | Average Error: 0.264245
Epoch: 35       | Average Error: 0.242176
Epoch: 40       | Average Error: 0.224356
Epoch: 45       | Average Error: 0.209403
Epoch: 49       | Average Error: 0.198969
```

</details>

### Test Accuracy

```
601 / 750 correct predictions
Accuracy: 80.13%
```


---

## 🗂️ Project Structure

```
cpp-cnn-from-scratch/
├── src/
│   ├── main.cpp            # Training and testing pipeline
│   ├── ConvLayer.h/.cpp    # Convolution layer (forward & backward)
│   ├── MaxPoolLayer.h/.cpp # Max pooling layer (forward & backward)
│   ├── SimpleMLP.h/.cpp    # Fully connected layers (forward & backward)
│   └── fileReader.h/.cpp   # MNIST CSV data loader
├── scripts/
│   └── mnist_to_csv.py     # Converts MNIST binary to CSV
├── data/                   # MNIST dataset files (not included)
├── Makefile                # Build configuration
└── README.md
```

---

## ⚙️ Implementation Details

### Convolution Layer
- Applies a single 3×3 learnable filter via sliding window
- ReLU activation: `max(0, x)`
- Backpropagation computes filter gradients through cross-correlation of the input with upstream gradients

### Max Pooling Layer
- 2×2 window with stride 2
- Stores the indices of maximum values for gradient routing during backpropagation
- Only the "winning" pixels receive gradients during the backward pass

### Fully Connected Layers (MLP)
- Two-layer MLP with sigmoid activation
- Backpropagation uses the sigmoid derivative: `σ'(x) = σ(x)(1 − σ(x))`
- Returns `dInput` to enable end-to-end gradient flow back to the convolution layer

### Data Pipeline
- A Python script converts MNIST IDX binary files to CSV format
- The C++ file reader parses CSV, applies one-hot encoding for labels, and normalizes pixel values to [0, 1]

---

## 🚀 Getting Started

### Prerequisites

- **C++ Compiler** with C++14 support (g++)
- **Eigen** library (header-only linear algebra library)
- **Python 3** (for dataset preparation)
- **Make**

### Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/ttalhA0/CNN-From-Scratch-Cpp.git
    cd cpp-cnn-from-scratch
    ```

2. **Install Eigen** (if not already installed)
    ```bash
    # Ubuntu / Debian
    sudo apt-get install libeigen3-dev

    # macOS
    brew install eigen
    ```

3. **Download the MNIST dataset**

    Download the following files from [MNIST Database](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and place them in the `data/` directory:
    - `train-images.idx3-ubyte`
    - `train-labels.idx1-ubyte`
    - `t10k-images.idx3-ubyte`
    - `t10k-labels.idx1-ubyte`

4. **Prepare the dataset**
    ```bash
    python scripts/mnist_to_csv.py
    ```

5. **Build and run**
    ```bash
    make
    ./cnn_mnist
    ```

---

## 🛠️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.005 |
| Epochs | 50 |
| Filter Size | 3×3 |
| Pool Size | 2×2 |
| Hidden Nodes | 64 |
| Output Nodes | 10 |
| Loss Function | MSE |
| Training Samples | 3,000 |
| Test Samples | 750 |

---

## 📚 Dependencies

| Dependency | Purpose |
|------------|---------|
| [Eigen](https://libeigen.gitlab.io/) | Matrix operations and linear algebra |
| Python 3 | Dataset preprocessing (IDX → CSV) |

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
