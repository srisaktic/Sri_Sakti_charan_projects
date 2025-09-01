# 🧠 DCGAN on MNIST: Handwritten Digit Generation

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic handwritten digits based on the MNIST dataset. The primary goal is to analyze how architectural changes, activation functions, and hyperparameter tuning affect the clarity and quality of generated images.

---

## 📌 Project Overview

Generative Adversarial Networks (GANs) are a class of neural networks where two models — a **Generator** and a **Discriminator** — are trained in opposition. This project focuses on:

- Building a **baseline DCGAN model**
- Comparing **activation functions**: LeakyReLU vs ReLU
- Tuning **hyperparameters**: noise vector dimension, batch size, learning rate, momentum
- Evaluating image quality across multiple **epochs** (10, 50, 100)

---

## 🧪 Experimental Setup

### ✅ Baseline Model
- **Noise vector (z-dim)**: 100
- **Activation**: LeakyReLU (Generator & Discriminator)
- **Batch size**: 64
- **Learning rate**: 1e-4
- **Momentum**: 0.9

### 🧪 Experiment 1: Epoch Comparison

| Epoch | Observations |
|-------|--------------|
| 10    | Poor clarity, high noise |
| 50    | Clearer digits, reduced noise |
| 100   | High clarity, stable, realistic digits |

### 🧪 Experiment 2: Activation Function Test

| Activation | Observations |
|------------|--------------|
| LeakyReLU  | Good stability and structure |
| ReLU       | Sharper digits, reduced artifacts, smoother learning |

### 🧪 Experiment 3: Hyperparameter Tuning

| Parameter         | Baseline | Modified |
|-------------------|----------|----------|
| Noise Vector      | 100      | 200      |
| Batch Size        | 64       | 256      |
| Learning Rate     | 1e-4     | 1e-3     |
| Momentum          | 0.9      | 0.7      |

🔎 **Result**: Modified settings led to unstable outputs and noisier images. High learning rate and lower momentum caused overshooting during training.

---

## 📈 Results Summary

- **Best Performance**: Baseline with LeakyReLU at epoch 100
- **Sharpest Digits**: ReLU activation at epoch 50
- **Worst Performance**: Modified hyperparameters

📌 *Insight*: ReLU yields clearer digits, but LeakyReLU provides better stability. Increasing noise vector dimension or learning rate must be done carefully to avoid degraded image quality.

---

## 📚 References

1. Ian Goodfellow et al., *Generative Adversarial Networks*, 2014. [`GAN (Original Paper).pdf`]
2. GAN Surveys and Overviews (included in repo)
3. MNIST Dataset - [http://yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist)

---

## 🧑‍💻 Author

**Sri Sakticharan Nirmal Kumar**  
📧 srisakticharan789@gmail.com  
📍 NYIT | MS Data Science

---

## 📝 License

This project is intended for academic and learning purposes only.
