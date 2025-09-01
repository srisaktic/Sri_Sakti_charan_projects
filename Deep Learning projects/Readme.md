# 🧠 Deep Learning Projects: CIFAR-10 Classification & DCGAN Digit Generation

## 📁 Repository Overview

This repository contains two deep learning projects aimed at demonstrating skills in:
- CNN model architecture design and transfer learning.
- Generative Adversarial Networks (GANs) for image generation.

---

## 🚀 Project 1: CIFAR-10 Image Classification

### 🔍 Objective
Classify 10 categories of images (airplane, car, bird, etc.) using:
- A custom-built CNN (`SimCNN`)
- A pre-trained deep CNN (`ResNet-50`) via transfer learning

### ✅ Techniques Used
- Data Augmentation (Random Flip, Crop, Normalization)
- Model Evaluation: Accuracy, Loss curves
- Comparison: Batch size impact (32 vs 64)
- Transfer Learning: ResNet50 with frozen + fine-tuned layers

### 📊 Results Summary
| Model       | Test Accuracy | Test Loss |
|-------------|---------------|-----------|
| SimCNN      | ~58%          | Moderate  |
| ResNet50    | ~70–73%       | Low       |

> ResNet50 outperformed SimCNN in terms of both accuracy and generalization.

---

## 🎨 Project 2: DCGAN on MNIST

### 🔍 Objective
Generate realistic handwritten digit images (0–9) using a **Deep Convolutional GAN (DCGAN)**.

### 🧠 Architecture
- **Generator**: Uses transposed convolutions to upscale noise vectors (latent space) into 28x28 digit images.
- **Discriminator**: Uses CNN layers to distinguish real vs. fake MNIST images.
- **Loss Function**: Binary Cross-Entropy (with adversarial training)
- **Optimizer**: Adam

### 📊 Results
- Generator was able to produce **visually coherent digits** after ~50 epochs.
- Learned to fool the discriminator, reaching a near 50/50 probability (GAN equilibrium).

> Demonstrates understanding of **unsupervised learning** and **generative modeling**.

---

## 🧰 Technologies Used

- Python 🐍
- PyTorch / TensorFlow (as used per script)
- Jupyter Notebooks
- Matplotlib, NumPy
- Deep Learning Concepts (CNNs, GANs, Transfer Learning)

---

## 📚 Learning Outcomes

| Skill Area               | Description |
|--------------------------|-------------|
| 🧱 CNN Design            | Learned to build and tune a ConvNet from scratch |
| ♻️ Transfer Learning     | Fine-tuned ResNet50 on CIFAR-10 dataset |
| 🎨 Generative Modeling   | Built a working GAN and trained it on MNIST |
| 📊 Model Evaluation       | Compared batch size, architecture performance, loss trends |
| 🧪 Experimentation        | Ran controlled experiments with clear hyperparameter changes |

---

## 📌 Conclusion

These projects highlight proficiency in:
- Building, training, and evaluating deep learning models.
- Experimenting with architecture variations and documenting results.
- Implementing real-world AI use cases (classification + image generation).

---

