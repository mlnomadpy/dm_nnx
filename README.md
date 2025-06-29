# The Dark Matter of Intelligence: Practical Workshops in Self-Supervised, Contrastive, and Multimodal Learning with Flax/nnx and JAX

This repository contains the code and practical workshops for the book "The Dark Matter of Intelligence".

## About the Book

"The Dark Matter of Intelligence" provides a hands-on approach to understanding and implementing cutting-edge machine learning techniques. Through a series of practical workshops, you will explore:

*   **Self-Supervised Learning:** Learn how models can learn from unlabeled data.
*   **Contrastive Learning:** Dive into techniques like SimCLR and BYOL to learn meaningful representations.
*   **Multimodal Learning:** Discover how to build models that can process and learn from multiple data types (e.g., images and text).

All examples and workshops are implemented using **JAX**, **Flax**, and the new **NNX** library, offering a modern and highly performant framework for deep learning research.

## Repository Structure

This repository is organized into several directories, each corresponding to different concepts and models discussed in the book.

*   `src/byol`: Implementation of Bootstrap Your Own Latent (BYOL).
*   `src/simclr`: Implementation of a Simple Framework for Contrastive Learning of Visual Representations (SimCLR).
*   `src/common`: Shared modules for data loading, training loops, and analysis.
*   `src/models`: Various model architectures, including ResNet and custom CNNs.

## Upcoming Workshops

We are continuously working on adding new content and workshops. Here are some of the topics and models that will be added in the future:

*   **Contrastive Learning Methods:**
    *   Momentum Contrast (MoCo)
    *   Swapping Assignments between multiple Views (SwAV)
    *   Self-DIstillation with NO labels (DINO)
*   **Multimodal Models:**
    *   Contrastive Language-Image Pre-training (CLIP)
*   **Generative Self-Supervised Methods:**
    *   Masked Autoencoders (MAE)

## Prerequisites

Before getting started, ensure you have:

- Python 3.9 or higher
- CUDA-compatible GPU (recommended for training)
- Git

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/dm_nnx.git
   cd dm_nnx
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Running SimCLR Training

```python
from src.simclr.main import main
main()
```

### Running BYOL Training

```python
from src.byol.main import main
main()
```

### Custom Dataset Training

You can train on your own image datasets by organizing them in the following structure:
```
your_dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

Then run:
```python
from src.byol.main import run_training_and_analysis

run_training_and_analysis(
    dataset_name="path/to/your_dataset",
    dataset_configs=your_configs,
    fallback_configs=fallback_configs,
    learning_rate=1e-3,
    momentum=0.9,
    pretrain=True  # Enable contrastive pretraining
)
```

## Workshop Highlights

### 🔬 Self-Supervised Learning Techniques
- **SimCLR**: Learn visual representations through contrastive learning
- **BYOL**: Bootstrap your own latent representations without negative sampling
- **Custom architectures**: YatCNN and ResNet implementations optimized for JAX/NNX

### 📊 Advanced Features
- **Flexible data loading**: Support for both TensorFlow Datasets and custom image folders
- **Comprehensive analysis**: Training curves, confusion matrices, and per-class accuracy
- **Modular design**: Easy to extend and modify for your own research
- **GPU acceleration**: Optimized for modern GPU training with JAX

### 🛠️ Implementation Details
- Built with the latest **Flax NNX** API for cleaner, more Pythonic neural network code
- Efficient data augmentation pipelines using TensorFlow
- Advanced loss functions including NT-Xent and BYOL regression loss
- Exponential moving averages for target networks
- Learning rate scheduling with cosine decay

## Key Features

| Feature | SimCLR | BYOL | Status |
|---------|--------|------|--------|
| Contrastive Pretraining | ✅ | ✅ | Complete |
| Fine-tuning | ✅ | ✅ | Complete |
| Custom Datasets | ✅ | ✅ | Complete |
| Analysis Tools | ✅ | ✅ | Complete |
| Checkpointing | ✅ | ✅ | Complete |

## Project Structure Explained

```
dm_nnx/
├── src/
│   ├── byol/                    # Bootstrap Your Own Latent implementation
│   │   ├── main.py             # Training pipeline and demo
│   │   └── model.py            # BYOL model with EMA target network
│   ├── simclr/                 # SimCLR implementation
│   │   └── main.py             # SimCLR training and analysis
│   ├── common/                 # Shared utilities
│   │   ├── analysis.py         # Evaluation and visualization tools
│   │   ├── data.py             # Data loading and preprocessing
│   │   ├── losses.py           # Loss functions (NT-Xent, BYOL, etc.)
│   │   └── train.py            # Training loops and optimization
│   └── models/                 # Model architectures
│       ├── __init__.py         # Model registry
│       ├── models.py           # Model factory and registration
│       ├── resnet.py           # ResNet implementation
│       └── yatcnn.py           # Custom CNN architecture
└── README.md
```

## Performance Tips

- **Batch Size**: Start with batch sizes of 256-512 for contrastive pretraining
- **Learning Rate**: Use learning rates around 1e-3 to 3e-4 for most datasets
- **Pretraining**: Run 10-50 epochs of contrastive pretraining before fine-tuning
- **Data Augmentation**: Strong augmentations are crucial for contrastive learning success
- **Hardware**: Multi-GPU training is supported through JAX's built-in parallelization

## Contributing

We welcome contributions to improve the workshops and add new techniques! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-technique`
3. **Make your changes** and ensure they follow the existing code style
4. **Add tests** for new functionality
5. **Submit a pull request** with a clear description of your changes

### Areas for Contribution
- New self-supervised learning methods
- Additional model architectures
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## Citation

If you use this code in your research, please cite:

```bibtex
@book{dark_matter_intelligence_2025,
  title={The Dark Matter of Intelligence: Practical Workshops in Self-Supervised, Contrastive, and Multimodal Learning with Flax/nnx and JAX},
  author={[Taha Bouhsine]},
  year={2025},
  publisher={[Publisher]}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The JAX and Flax teams for providing excellent deep learning frameworks
- The original authors of SimCLR, BYOL, and other techniques implemented here
- The open-source community for inspiration and feedback

## Support

If you encounter any issues or have questions:

1. **Check the Issues**: Look through existing [GitHub Issues](https://github.com/yourusername/dm_nnx/issues)
2. **Create a New Issue**: If your problem isn't covered, create a new issue with:
   - A clear description of the problem
   - Steps to reproduce
   - Your environment details (OS, Python version, GPU, etc.)
   - Any error messages

## Roadmap

- [ ] Add MoCo implementation
- [ ] Implement SwAV technique
- [ ] Add comprehensive unit tests
- [ ] Create Jupyter notebook tutorials
- [ ] CLIP implementation for multimodal learning
- [ ] Masked Autoencoder (MAE) workshop
- [ ] Multi-GPU training examples
- [ ] Docker containerization
- [ ] DINO self-distillation method
- [ ] Advanced augmentation strategies
- [ ] Distributed training across multiple nodes
- [ ] Integration with MLflow for experiment tracking

---

**Happy Learning! 🚀**

*For the latest updates and announcements, follow the repository and star it if you find it useful!*
