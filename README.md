# 🧠 CNN From Scratch in C++

A **Convolutional Neural Network** built entirely from scratch in C++ without any deep learning framework. Trained and tested on the **MNIST** handwritten digit dataset.

The goal of this project is to understand the inner workings of CNNs by implementing every layer — convolution, pooling, and fully connected — along with their forward and backward passes manually.

---

## 📐 Network Architecture

```
Input (28×28) → Conv2D (32×3×3, ReLU) → MaxPool (2×2) → Flatten → Dense (5408→128, Sigmoid) → Dense (128→10, Sigmoid) → Output
```

| Layer | Input | Output | Parameters | Activation |
|-------|-------|--------|------------|------------|
| Convolution | 28×28 | 32 × 26×26 | 320 | ReLU |
| Max Pooling | 32 × 26×26 | 32 × 13×13 | 0 | — |
| Flatten | 32 × 13×13 | 5,408 | 0 | — |
| Hidden (FC) | 5,408 | 128 | 692,352 | Sigmoid |
| Output (FC) | 128 | 10 | 1,290 | Sigmoid |
| **Total** | | | **693,962** | |

---

## 📊 Training Results

Trained on **3,000** images for **50 epochs** with a learning rate of **0.01**.

<details>
<summary>Training Log (click to expand)</summary>

```
Epoch: 0        | Average Error: 1.06051
Epoch: 1        | Average Error: 0.776639
Epoch: 2        | Average Error: 0.688637
Epoch: 3        | Average Error: 0.602865
Epoch: 4        | Average Error: 0.534134
Epoch: 5        | Average Error: 0.475726
Epoch: 10       | Average Error: 0.308839
Epoch: 15       | Average Error: 0.227701
Epoch: 20       | Average Error: 0.183630
Epoch: 25       | Average Error: 0.154389
Epoch: 30       | Average Error: 0.132493
Epoch: 35       | Average Error: 0.120169
Epoch: 40       | Average Error: 0.109045
Epoch: 45       | Average Error: 0.099879
Epoch: 49       | Average Error: 0.094796
```

</details>

### Test Accuracy

```
649 / 750 correct predictions
Accuracy: 86.53%
```


---

## 🗂️ Project Structure

```
cpp-cnn-from-scratch/
├── src/
│   ├── train.cpp           # Training pipeline (saves weights to .bin)
│   ├── inference.cpp       # Inference pipeline (loads weights from .bin)
│   ├── ConfigParser.h      # Custom parser for reading hyperparameter config
│   ├── ConvLayer.h/.cpp    # Convolution layer (forward & backward)
│   ├── MaxPoolLayer.h/.cpp # Max pooling layer (forward & backward)
│   ├── SimpleMLP.h/.cpp    # Fully connected layers (forward & backward)
│   └── fileReader.h/.cpp   # MNIST CSV data loader
├── scripts/
│   └── mnist_to_csv.py     # Converts MNIST binary to CSV
├── data/                   # MNIST dataset files (not included)
├── config.txt              # Dynamically loaded hyperparameters
├── Makefile                # Build configuration
└── README.md
```

---

## ⚙️ Implementation Details

### Convolution Layer
- Applies **32 independent 3×3 learnable filters**, each producing a separate feature map
- ReLU activation: `max(0, x)`
- Backpropagation computes per-filter gradients through cross-correlation of the input with upstream gradients

### Max Pooling Layer
- 2×2 window with stride 2, applied independently to each of the 32 feature maps
- Stores the indices of maximum values per channel for gradient routing during backpropagation
- Only the "winning" pixels receive gradients during the backward pass

### Fully Connected Layers (MLP)
- Two-layer MLP with sigmoid activation
- Backpropagation uses the sigmoid derivative: `σ'(x) = σ(x)(1 − σ(x))`
- Returns `dInput` to enable end-to-end gradient flow back to the convolution layer

### Data Pipeline
- A Python script converts MNIST IDX binary files to CSV format
- The C++ file reader parses CSV, applies one-hot encoding for labels, and normalizes pixel values to [0, 1]

### Production-Ready Pipeline
- **Decoupled Architecture:** The system is split into two independent executables: `train` and `inference`.
- **Weight I/O:** Model parameters are securely saved and loaded as binary blocks (`.bin`) using standard C++ I/O streams, ensuring rapid memory allocation without overhead.
- **Dynamic Configuration:** Hyperparameters are parsed dynamically from `config.txt` at runtime. The Flatten layer size is automatically computed, completely eliminating the need for recompilation during experimentation.

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

5. **Build the executables**
    ```bash
    make
    ```

6. **Run Training**
    ```bash
    # Trains the model using config.txt and saves weights to .bin files
    ./train
    
    # Or specify a custom dataset path:
    ./train -d data/custom_training_set.csv
    ```

7. **Run Inference**
    ```bash
    # Loads .bin weights and evaluates the network on the test set
    ./inference

    # Or specify a custom test dataset path:
    ./inference -d data/custom_test_set.csv
    ```
---

## 🛠️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.01 |
| Epochs | 50 |
| Number of Filters | 32 |
| Filter Size | 3×3 |
| Pool Size | 2×2 |
| Hidden Nodes | 128 |
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
