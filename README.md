# Neural Network from Scratch - MNIST Digit Classifier

A complete neural network implementation from scratch using **NumPy**, trained on the **MNIST** dataset for handwritten digit classification. This project does not use any ML libraries like TensorFlow or PyTorch , just raw matrix math, backpropagation, and optimization.

---

## Dataset

- **MNIST** dataset: 70,000 images (60,000 train + 10,000 test)
- Grayscale 28x28 pixel handwritten digits (0–9)
- Flattened to 784-dimensional vectors

Data source: [`fetch_openml('mnist_784')`](https://www.openml.org/d/554)

---

## Model Architecture

```
Input:   784 nodes
Hidden1: 128 neurons + ReLU + Dropout
Hidden2: 64 neurons + ReLU + Dropout
Output:  10 neurons + Softmax
```

<img width="1923" height="1082" alt="image" src="https://github.com/user-attachments/assets/06a9ca80-3b22-4f5e-bc49-9821189a161e" />



---

## Data Preparation

- Normalized pixel values to [0, 1] by dividing by 255
- One-hot encoded labels for output layer
- Split into training and test sets
- Visualized samples using matplotlib


  <img width="739" height="324" alt="image" src="https://github.com/user-attachments/assets/96b12d1b-acf1-43e4-89c1-5b3585e37a33" />


---

## Feedforward Process

1. Multiply input with weights + biases for Layer 1
2. Apply **ReLU activation**
3. Apply **Dropout** (only during training)
4. Repeat for Layer 2
5. Output layer uses **Softmax** to produce probabilities

---

## Backpropagation

- Computed gradients of cross-entropy loss w.r.t. weights and biases
- Applied chain rule to propagate error backward through the network
- Implemented updates using:
  - SGD
  - SGD with Momentum
  - Adam
- Included **L2 regularization** in loss and weight updates

---

## Training and Evaluation

- Trained over multiple epochs using mini-batches
- Tracked both training and validation loss/accuracy
- Evaluated final model on test set
- Plotted learning curves

---

## Best Results (With L2 + Dropout)

| Optimizer         | Train Acc | Val Acc | Val Loss |
|------------------|-----------|---------|----------|
| SGD              | 100%      | **97.6%**   | 0.1093   |
| SGD + Momentum   | 100%      | 97.57%  | 0.1126   |
| Adam             | 100%      | 97.53%  | 0.1161   |

---

## Highlights

- Neural net built using only NumPy
- Fully working feedforward and backpropagation
- Regularization via **Dropout** and **L2**
- Supports multiple optimizers
- Modular and easy to extend

---

 Built with math, loops, and coffee ☕
