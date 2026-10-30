# fys-stk4155_project2

## TODO

### Part a)

#### Backend
- ~~Write python functions for cost, with gradients~~.
- ~~Implement Sigmoid, ReLU and leaky ReLU functions with derivatives.~~

#### Frontend

#### Report
- Derive expressions for the cost/loss functions with gradients of:
  - MSE with and without $L_1$ and $L_2$ norms (regression problems)
  - Binary cross entropy (log loss) with and without $L_1$ and $L_2$ norms (binary classification problems)
  - Multiclass cross entropy cost/loss function (softmax cross entropy)
- Set up above three cost/loss functions and respective derivatives and explain the various terms.
- Set up expressions and first derivatives of:
  - Sigmoid function
  - ReLU function
  - Leaky ReLU function

### Part b)

#### Backend
- ~~Implement Runge function~~
- ~~Write feed forward neural network code for regression problem with number of layers, nodes and activation functions as parameters. Initialize weights using normal distribution. Initialize biases using a small non-zero positive value.~~
- ~~Implement back propagation algorithm for computing gradient of neural network.~~
- Make gradient descent methods work with neural network cost function.

#### Frontend
- Train feed forward neural network on Runge function, and compare with OLS regression from project 1. Use single input for x-value, and single output for y-value, with sigmoid on the last layer. Can possibly compare with using ReLU on the last layer as well. When comparing with project 1, use the data sets and polynomial order which gave the best MSE score in project 1. Compare using a neural network with one and two hidden layers using 50 and 100 nodes, respectively. Test stochastic and non-stochastic gradient descent with RMSprop, Adam, and plain gradient descent with different learning rates.
- Remember to scale data.

#### Report
- Comment on results and give critical discussion of the results obtained with the OLS code from project 1 and our own neural network code. Make analysis of the learning rates employed to find the optimal MSE score.

### Part c)

#### Backend

#### Frontend
- Test different activation functions for hidden layers (Sigmoid, ReLU, leaky ReLU).
- Test results as functions of number of hidden layers and nodes.

#### Report
- Discuss results of activation functions, and number of hidden layers and nodes.
- Discuss the effect of vanishing gradients when using sigmoid on hidden layers in deep networks.
- Perform bias-variance trade-off analysis (do we see signs of overfitting?)

### Part d)

#### Backend
- ~~Implement ridge and lasso penalties with hyperparameters $\lambda$.~~

#### Frontend
- Find optimal results for hyperparameters $\lambda$ and learning rates $\gamma$ and neural network architecture and compare $L_2$ results with ridge regression and $L_1$ results with lasso from project 1. Use the same data sets for the comparisons.

#### Report
- Discuss differences between ridge/lasso in neural networks vs ridge/lasso in linear regression.

### Part e)

#### Backend
- ~~Implement logistic regression for classification problems.~~ This one is technically already done since logistic regression is the same as a neural network with no hidden layers and sigmoid activation function.

#### Frontend
- Load MNIST data set ($70 \ 000$ images of $28 \times 28$ pixels) ($60 \ 000$ training images and $10 \ 000$ test images).
- Rescale the pixel values to the interval $[0, 1]$.
- All training, model selection and tuning of hyperparameters is done on the training data only (use validation set or cross-validation).
- Measure performance using accuracy score, $\text{Accuracy} = \frac{\Sigma_{i=1}^{n} I(t_i = y_i)}{n}$.
- Compare neural network code for classification and results against similar code using scikit-learn or TensorFlor/Keras or PyTorch.
- Compare neural network results with those from logistic regression.
- Compute confusion matrix of best classification model, as well as precision, recall and $F_1$ score.

#### Report
- Cite the data set (MNIST) in report.
- "Feel free to suggest other data sets."
- Discuss results and give a critical analysis of the various parameters, including hyperparameters like learning rates and regularization parameter $\lambda$, various activation functions, number of hidden layers and nodes and activation functions.
- Interpret confusion matrix of best classification model, along with precision, recall and $F_1$ score.

### Part f)

#### Backend

#### Frontend

#### Report
- Summarize the various algorithms and come with a critical evaluations of pros and cons. Which algorithm works best for regression case and which for the classification case?
