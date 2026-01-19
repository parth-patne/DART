<p align="center">
  <img src="https://img.shields.io/badge/ICCAI_2025-Accepted-success?style=for-the-badge" alt="ICCAI 2025 Accepted">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License">
</p>

<h1 align="center">🎯 DART: Dynamic Adaptive Real-time Thresholding for Early-Exit DNNs</h1>

<p align="center">
  <strong>Input-Difficulty-Aware Adaptive Threshold Mechanism for Efficient Deep Neural Network Inference</strong>
</p>

<p align="center">
  <em>Accepted at the <strong>International Conference on Communications and Artificial Intelligence (ICCAI 2025)</strong></em>
</p>

---

## 📖 Overview

**DART** introduces a novel approach to early-exit deep neural networks by dynamically adjusting confidence thresholds based on input difficulty. Unlike traditional static threshold methods, DART computes a **difficulty score (α)** for each input sample and uses reinforcement learning to optimize exit decisions, achieving superior accuracy-efficiency trade-offs.

### 🔑 Key Contributions

1. **Difficulty-Aware Threshold Adaptation**: A lightweight preprocessing module computes input difficulty using edge density, pixel variance, and gradient complexity metrics.

2. **Reinforcement Learning-Based Exit Policy**: Q-Learning agents learn optimal threshold adjustments based on input characteristics and computational constraints.

3. **Joint Exit Optimization**: A dynamic programming-based policy optimizes exit decisions across all exit points simultaneously, considering global computational costs.

4. **Cost-Aware Inference**: Incorporates real-time computation and energy cost awareness for resource-constrained edge deployment.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DART Framework                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐   │
│  │ Input Image (x)  │────▶│ Difficulty        │────▶│ α ∈ [0, 1]       │   │
│  └──────────────────┘     │ Estimator         │     │ (Difficulty      │   │
│                           │ ┌───────────────┐ │     │  Score)          │   │
│                           │ │ Edge Density  │ │     └────────┬─────────┘   │
│                           │ │ Pixel Variance│ │              │             │
│                           │ │ Gradient      │ │              │             │
│                           │ │ Complexity    │ │              ▼             │
│                           │ └───────────────┘ │     ┌──────────────────┐   │
│                           └───────────────────┘     │ Adaptive         │   │
│                                                     │ Threshold        │   │
│  ┌─────────────────────────────────────────────┐   │ τ(α, i)          │   │
│  │        Backbone Network with Exits          │   └────────┬─────────┘   │
│  │                                             │            │             │
│  │  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐    │            │             │
│  │  │Exit │   │Exit │   │Exit │   │Exit │    │◀───────────┘             │
│  │  │  1  │   │  2  │   │ ... │   │  N  │    │                          │
│  │  └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘    │                          │
│  │     │         │         │         │       │                          │
│  │  [Conf₁]   [Conf₂]   [Conf₃]   [ConfN]    │                          │
│  └─────┼─────────┼─────────┼─────────┼───────┘                          │
│        │         │         │         │                                   │
│        ▼         ▼         ▼         ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              Joint Exit Policy (Q-Learning / DP)                  │   │
│  │  • State: (α_bin, exit_idx, confidence_bin)                      │   │
│  │  • Action: EXIT or CONTINUE                                       │   │
│  │  • Reward: Accuracy + Efficiency + Difficulty Adjustment          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
DART/
├── 📄 README.md                         # This file
├── 📄 LICENSE                           # MIT License
├── 📄 DART.pdf                          # Research paper
│
├── 🐍 dynamic-alexnet-early-exit.py     # DART implementation for AlexNet
│   └── Full implementation with 6 exit points, difficulty estimation,
│       Q-Learning agents, and comprehensive evaluation
│
├── 🐍 dynamic-resnet18-early-exits.py   # DART implementation for ResNet-18
│   └── 5 exit points adapted for residual connections
│
├── 🐍 dynamic-vgg-early-exits.py        # DART implementation for VGG
│   └── 6 exit points for deeper VGG architecture
│
└── 📁 LeViT-main/                       # Vision Transformer adaptations
    ├── enhanced_adaptive_levit.py       # DART-enhanced LeViT with 4 exits
    ├── enhanced_levit_training.py       # Training scripts for LeViT
    ├── final_clean_comparison.py        # Baseline comparison utilities
    ├── levit.py                         # Original LeViT implementation
    ├── datasets.py                      # Dataset loading utilities
    ├── engine.py                        # Training engine
    └── utils.py                         # Helper functions
```

---

## 🧠 Core Components

### 1. Difficulty Estimator (`DifficultyEstimator`)

A lightweight preprocessing module that computes a difficulty score α ∈ [0, 1] for each input:

```python
class DifficultyEstimator(nn.Module):
    def __init__(self, w1=0.4, w2=0.3, w3=0.3):
        # Combines three metrics:
        # - Edge Density (Sobel operator)
        # - Pixel Variance
        # - Gradient Complexity (Laplacian)
```

**Features:**
- **Zero training overhead**: Uses fixed operators (Sobel, Laplacian)
- **GPU-accelerated**: Fully differentiable convolution operations
- **Batch processing**: Efficient vectorized computation

### 2. Threshold Q-Learning Agent (`ThresholdQLearningAgent`)

Learns optimal threshold coefficient adjustments:

```python
class ThresholdQLearningAgent:
    def __init__(self, n_exits, alpha_bins=10, epsilon=0.1, alpha=0.1, gamma=0.9):
        # State: (α_bin, exit_idx)
        # Actions: [-0.1, 0.0, +0.1] coefficient adjustments
```

### 3. Joint Exit Policy (`JointExitPolicy`)

Dynamic programming-based policy for global optimization:

```python
class JointExitPolicy:
    def __init__(self, n_exits, alpha_bins=20, confidence_bins=20, gamma=0.95):
        # Value iteration for optimal exit policy
        # Considers computation costs across all exits
```

### 4. Early Exit Architectures

| Architecture | Exit Points | Parameters | Supported Datasets |
|-------------|-------------|------------|-------------------|
| **AlexNet** | 6 exits | ~2.5M | MNIST, CIFAR-10/100 |
| **ResNet-18** | 5 exits | ~11M | CIFAR-10/100, ImageNet |
| **VGG-16** | 6 exits | ~15M | CIFAR-10/100, ImageNet |
| **LeViT** | 4 exits | ~7-39M | ImageNet |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/DART.git
cd DART

# Create conda environment
conda create -n dart python=3.8 -y
conda activate dart

# Install dependencies
pip install torch torchvision
pip install numpy matplotlib scikit-learn seaborn pandas
pip install pynvml  # For GPU monitoring (optional)
```

### Training AlexNet with DART on CIFAR-10

```python
import torch
from dynamic-alexnet-early-exit import FullLayerAlexNet, DifficultyEstimator

# Initialize model with all DART features
model = FullLayerAlexNet(
    num_classes=10,
    use_difficulty_scaling=True,    # Enable α computation
    use_joint_policy=True,          # Enable DP-based exit policy
    use_cost_awareness=True         # Enable cost-aware inference
)

# Training mode
model.training_mode = True
outputs = model(images)  # Returns outputs from all 6 exits

# Inference mode
model.training_mode = False
predictions, exit_points, costs = model(images)
```

### Evaluating Exit Distribution

```python
# Get analysis data after inference
analysis = model.get_analysis_data()

# Alpha distribution
print(f"Mean difficulty: {np.mean(analysis['alpha_values']):.3f}")

# Exit point distribution
exit_counts = [0] * 6
for log in analysis['exit_decisions_log']:
    exit_counts[log['exit_point'] - 1] += len(log['sample_indices'])
print(f"Exit distribution: {exit_counts}")
```

---

## 📊 Experimental Results

### Accuracy vs. Computational Savings

| Method | CIFAR-10 Acc. | FLOPs Reduction | Throughput Gain |
|--------|---------------|-----------------|-----------------|
| Baseline (No EE) | 91.2% | 0% | 1.0× |
| Static Threshold | 89.8% | 35% | 1.4× |
| BranchyNet | 90.1% | 32% | 1.3× |
| **DART (Ours)** | **90.9%** | **45%** | **1.7×** |

### Exit Distribution Analysis

DART adaptively routes samples based on difficulty:

| Difficulty (α) | Exit 1-2 | Exit 3-4 | Exit 5-6 |
|---------------|----------|----------|----------|
| Easy (α < 0.3) | 65% | 25% | 10% |
| Medium (0.3 ≤ α < 0.7) | 20% | 55% | 25% |
| Hard (α ≥ 0.7) | 5% | 20% | 75% |

---

## 🔧 Configuration Options

### Model Initialization

```python
model = FullLayerAlexNet(
    num_classes=10,                 # Number of output classes
    in_channels=3,                  # Input channels (1 for MNIST, 3 for CIFAR)
    use_difficulty_scaling=True,    # Enable difficulty-aware thresholds
    use_joint_policy=True,          # Enable joint exit optimization
    use_cost_awareness=True         # Enable cost-aware decisions
)
```

### Difficulty Estimator Weights

```python
difficulty_estimator = DifficultyEstimator(
    w1=0.4,  # Edge density weight
    w2=0.3,  # Pixel variance weight
    w3=0.3   # Gradient complexity weight
)
```

### Q-Learning Hyperparameters

```python
agent = ThresholdQLearningAgent(
    n_exits=6,        # Number of exit points
    alpha_bins=10,    # Discretization bins for α
    epsilon=0.1,      # Exploration rate
    alpha=0.1,        # Learning rate
    gamma=0.9         # Discount factor
)
```

---

## 📈 Visualization

The codebase includes comprehensive visualization utilities:

- **Confusion matrices** per exit point
- **Exit distribution histograms**
- **Alpha (difficulty) distribution plots**
- **Accuracy vs. computation trade-off curves**
- **Real-time GPU monitoring**

---

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{dart2025iccai,
  title={DART: Input-Difficulty-Aware Adaptive Threshold for Early-Exit DNNs},
  author={Patne, Parth},
  booktitle={Proceedings of the International Conference on Communications and Artificial Intelligence (ICCAI)},
  year={2025}
}
```

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- LeViT implementation adapted from [Facebook Research](https://github.com/facebookresearch/LeViT)
- Inspired by early-exit works including BranchyNet, EENet, and RACENet
- GPU monitoring utilities powered by NVIDIA's pynvml

---

<p align="center">
  <strong>⭐ Star this repository if you find it helpful!</strong>
</p>
