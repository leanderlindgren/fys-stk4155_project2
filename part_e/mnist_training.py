import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append("../")

import numpy as np
np.set_printoptions(suppress=True)

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.linear_model import LogisticRegression

from functions.activation_functions import sigmoid, sigmoid_der, softmax
from functions.cost_functions import cross_entropy, cross_entropy_Lp
from models.neural_network import NeuralNetwork


def load_mnist_rows():
    with np.load("part_e/mnist.npz") as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) / 255.0

    return x_train, y_train, x_test, y_test


def mnist_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def mnist_log_ref(xtr, xte, ytr, yte, lambdas):

    log_acc = []

    for lmbda in lambdas:

        print(f"Starting lambda = {lmbda:.1e}")

        model = LogisticRegression(
            C=1/lmbda,
            solver="saga",
            max_iter=100,
            random_state=42,
            verbose=1
        )

        model.fit(xtr, ytr)
        print("Model Fit")

        pred = model.predict(xte)
        print("Predict")

        log_acc.append(np.mean(pred == yte))

    return log_acc


# --------------------------------------------------
# Load data
# --------------------------------------------------

randnum = 2026
np.random.seed(randnum)

x_train, y_train, x_test, y_test = load_mnist_rows()

y_train_onehot = np.eye(10)[y_train]
y_test_onehot = np.eye(10)[y_test]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# --------------------------------------------------
# Neural network
# --------------------------------------------------

nn = NeuralNetwork(
    network_input_size=784,
    layer_output_sizes=[100, 10],
    activation_funcs=[sigmoid, softmax],
    activation_ders=[sigmoid_der, None],
    cost_fun=cross_entropy,
    cost_der=cross_entropy_Lp
)

nn.initialize_layers()

epochs = 100
gamma = 0.01

losses = []

for epoch in range(epochs):

    grads = nn.compute_gradient(
        x_train,
        y_train_onehot
    )

    nn.update_weights(
        grads,
        gamma
    )

    loss = nn.cost(
        x_train,
        y_train_onehot
    )

    losses.append(loss)

    print(epoch, loss)

# --------------------------------------------------
# Evaluate NN
# --------------------------------------------------

pred = nn.predict(x_test)

pred_labels = np.argmax(pred, axis=1)

accuracy = np.mean(pred_labels == y_test)

print("Test Accuracy =", accuracy)

cm = confusion_matrix(y_test, pred_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.savefig("results/mnist_confusion_matrix.png", dpi=300)
plt.show()

print("\nClassification Report (Precision, Recall, F1-Score):")
print(
    classification_report(
        y_test,
        pred_labels,
        digits=4
    )
)

# --------------------------------------------------
# Loss curve
# --------------------------------------------------

plt.figure(figsize=(8, 6))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Neural Network Training Loss")
plt.grid(True)
plt.savefig("results/mnist_loss.png", dpi=300)
plt.show()

# --------------------------------------------------
# Logistic Regression comparison
# --------------------------------------------------

lambdas = np.logspace(-5, -1, 5)

print("Starting logistic regression")

log_acc = mnist_log_ref(
    x_train,
    x_test,
    y_train,
    y_test,
    lambdas
)

print("Finished logistic regression")

plt.figure(figsize=(6, 4))
plt.semilogx(lambdas, log_acc, "o-")
plt.xlabel(r"$\lambda$")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Accuracy on MNIST")
plt.savefig("results/mnist_logistic_accuracy.png", dpi=300)
plt.show()