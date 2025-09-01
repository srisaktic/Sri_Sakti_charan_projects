# 🧠 CIFAR-10 Image Classification using SimCNN and ResNet50

This deep learning project focuses on image classification using the **CIFAR-10 dataset**, comparing a custom-built CNN architecture (**SimCNN**) with a pretrained **ResNet50** model. The goal is to evaluate performance, training efficiency, and generalization capability across models and hyperparameter configurations.

---

## 🧾 Dataset

- **Name**: CIFAR-10
- **Type**: Image Classification
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image Size**: 32x32 pixels
- **Training Samples**: 50,000
- **Test Samples**: 10,000

---

## 🔧 Models & Architectures

### 1. **SimCNN (Custom CNN)**

- 3 Convolutional + MaxPooling layers
- Flatten + Dense layers
- Dropout for regularization
- **Batch Size**: 32
- **Epochs**: 25
- **Optimizer**: Adam
- **Performance**:
  - Test Accuracy: **68.75%**
  - Test Loss: **0.9260**

---

### 2. **ResNet50 (Pretrained on ImageNet)**

#### (a) Default Parameters
- **Batch Size**: 32  
- **Epochs**: 25  
- **Steps per Epoch**: 550  
- **Test Accuracy**: **95.93%**  
- **Test Loss**: **0.1683**

#### (b) Custom Parameters (ResNet50-02)
- **Batch Size**: 64  
- **Epochs**: 30  
- **Test Accuracy**: **94.67%**  
- **Test Loss**: **0.2565**

---

## 📊 Model Comparison

| Model         | Batch Size | Epochs | Test Accuracy | Test Loss |
|---------------|------------|--------|----------------|-----------|
| SimCNN        | 32         | 25     | 68.75%         | 0.9260    |
| ResNet50-01   | 32         | 25     | 95.93%         | 0.1683    |
| ResNet50-02   | 64         | 30     | 94.67%         | 0.2565    |

---

## 📌 Insights

- ✅ **ResNet50** significantly outperforms SimCNN in both accuracy and loss.
- ✅ Increasing the **batch size** and **epochs** helped speed up training (ResNet50-02), but didn't improve accuracy.
- ⚠️ SimCNN is limited in generalization and requires more training to achieve moderate accuracy.

---

## 📈 Justification for Custom Parameters

- **Batch Size = 64**: Speeds up training and stabilizes gradients.
- **Epochs = 30**: Allows deeper models like ResNet50 to learn richer features.
- **Learning Rate = 0.005**: Lowered for stable convergence without overshooting.

Steps per Epoch = `50000 / 64 ≈ 782`  
Ensures entire dataset is covered in every epoch without data loss.

---

## 🏁 Conclusion

- **Best Model**: ResNet50-01 (default parameters)  
- **Recommendation**: Use pretrained architectures like ResNet50 for small image datasets like CIFAR-10 when high accuracy is needed.

---

## 🚀 Future Work

- Try advanced models like EfficientNet or DenseNet.
- Apply transfer learning on different domains.
- Hyperparameter tuning via tools like Optuna or Grid Search.

---

## 📑 Authors

**Sri Sakticharan Nirmal Kumar**  
NYIT | Deep Learning | Fall 2024  
Student ID: 1337576  

---
