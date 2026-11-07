import sys
sys.path.append("../")
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.ols import OLS
from functions.activation_functions import sigmoid, sigmoid_der, softmax
from functions.cost_functions import mse, mse_der
from functions.runge import runge

class FFNN:
    def __init__(
        self,
        n_inputs,
        n_hidden,
        n_outputs,
        eta=0.01,
        lmbda=0.0
    ):

        self.eta = eta
        self.lmbda = lmbda

        # Input -> Hidden
        self.W1 = np.random.randn(n_inputs, n_hidden)
        self.b1 = np.zeros((1, n_hidden))

        # Hidden -> Output
        self.W2 = np.random.randn(n_hidden, n_outputs)
        self.b2 = np.zeros((1, n_outputs))

    def feed_forward(self, X):

            self.z1 = X @ self.W1 + self.b1
            self.a1 = sigmoid(self.z1)

            self.z2 = self.a1 @ self.W2 + self.b2
            self.a2 = softmax(self.z2)

            return self.a2

    def predict(self, X):

        outputs = self.feed_forward(X)

        return np.argmax(outputs, axis=1)

    def one_hot(y):
        return np.eye(10)[y]

    def fit(self, X, y, epochs=20):

        # one-hot encode labels
        y = np.eye(10)[y]

        for epoch in range(epochs):

            # ---------- feed forward ----------
            output = self.feed_forward(X)

            # ---------- output layer ----------
            #delta2 = mse_der(output, y) * softmax(self.z2)
            delta2 = output - y

            # ---------- hidden layer ----------
            delta1 = (delta2 @ self.W2.T) * sigmoid_der(self.z1)

            # ---------- gradients ----------
            n = X.shape[0]

            dW2 = self.a1.T @ delta2 / n + self.lmbda * self.W2
            db2 = np.sum(delta2, axis=0, keepdims=True) / n

            dW1 = X.T @ delta1 / n + self.lmbda * self.W1
            db1 = np.sum(delta1, axis=0, keepdims=True) / n


            # ---------- gradient descent ----------
            self.W2 -= self.eta * dW2
            self.b2 -= self.eta * db2

            self.W1 -= self.eta * dW1
            self.b1 -= self.eta * db1

            loss = mse(y, output)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {loss}")

def mnist_ffnn(x_train, y_train,
               x_test, y_test,
               eta, lmbda,
               epochs,
               batch_size,
               hidden_nodes):

    nn = FFNN(
        n_inputs=784,
        n_hidden=hidden_nodes,
        n_outputs=10,
        eta=eta,
        lmbda=lmbda
    )

    nn.fit(
        x_train,
        y_train,
        epochs=epochs
    )

    y_pred = nn.predict(x_test)

    acc = np.mean(y_pred == y_test)

    return acc, nn

def load_mnist_rows():
    with np.load("part_e//mnist.npz") as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"] 

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

    #feed forward network takes a vector input, 
    # so we flatten the vector of the input to 784 features 
    # and scale the pixel value to the vector |0,1|
    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) / 255.0

    return (x_train, y_train, x_test, y_test)

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


best_acc = 0
best_eta = None
best_lmbda = None
best_model = None

xtr, ytr, xte, yte = load_mnist_rows()
print(xtr.shape, ytr.shape, xte.shape, yte.shape)

nn = FFNN(
    n_inputs=784,
    n_hidden=50,
    n_outputs=10
)

pred = nn.predict(xte)

print(pred.shape)

accuracy = mnist_accuracy(yte, pred)
print(accuracy)

etas = np.logspace(-4, 0, 5)
lambdas = np.logspace(-5, -1, 5)

accuracy = np.zeros((len(etas), len(lambdas)))

for i, eta in enumerate(etas):
    for j, lmbda in enumerate(lambdas):

        acc, model = mnist_ffnn(
            xtr, ytr,
            xte, yte,
            eta=eta,
            lmbda=lmbda,
            epochs=100,
            batch_size=100,
            hidden_nodes=100
        )

        if acc > best_acc:
            best_acc = acc
            best_eta = eta
            best_lmbda = lmbda
            best_model = model

        accuracy[i, j] = acc

fig, ax = plt.subplots(figsize=(8,6))

im = ax.imshow(
    accuracy,
    cmap="viridis",
    origin="lower"
)

ax.set_xticks(range(len(lambdas)))
ax.set_xticklabels([f"{x:.0e}" for x in lambdas])

ax.set_yticks(range(len(etas)))
ax.set_yticklabels([f"{x:.0e}" for x in etas])

ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\eta$")
ax.set_title("MNIST Classification Accuracy")

plt.colorbar(im, label="Accuracy")

plt.tight_layout()
plt.savefig("results/mnist_accuracy_heatmap.png", dpi=300)

plt.show()

print("best accuracy", best_acc)
print("best eta", best_eta)
print("best lmbda", best_lmbda)

best_pred = best_model.predict(xte)

cm = confusion_matrix(yte, best_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("Confusion Matrix for Best FFNN")
plt.savefig("results/mnist_confusion_matrix.png", dpi=300)
plt.show()

print("Starting logistic regression")
log_acc = mnist_log_ref(xtr, xte, ytr, yte, lambdas)
print("Finished logistic regression")

plt.figure(figsize=(6,4))

plt.semilogx(lambdas, log_acc, "o-")

plt.xlabel(r"$\lambda$")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Accuracy on MNIST")

plt.savefig("results/mnist_logistic_accuracy.png", dpi=300)
plt.show()

print("\nClassification Report (Precision, Recall, F1-Score):")
print(classification_report(yte, best_pred, digits=4))