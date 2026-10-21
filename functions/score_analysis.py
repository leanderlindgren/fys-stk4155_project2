def MSE(x, y, seed n_bootstraps, maxdegree):

    #test_size 0.2 because we're reserving 20% of the data for testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        seed)

    error = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)

    degrees = range(maxdegree)

    #array to get the MSE
    mean_scores = []

    for degree in range(maxdegree):
        model = make_pipeline(PolynomialFeatures(degree=degree),
                            LinearRegression(fit_intercept=False))
        y_pred = np.empty((y_test.shape[0], n_bootstraps))
        for i in range(n_bootstraps):
            x_, y_ = resample(x_train, y_train)
            y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

        # Expectations over training sets are averages along axis 1;
        # keepdims=True preserves the column shape and is essential
        # for the bias to come out right.
        error[degree] = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
        bias[degree] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
        variance[degree] = np.mean(np.var(y_pred, axis=1, keepdims=True))

        # negative MSE, so we will invert sign later
        scores = cross_val_score(
            model, x_train, y_train,
            scoring="neg_mean_squared_error",
            cv=10
        )
        
        mean_scores.append(scores.mean())

    mean_scores = np.array(mean_scores)

    # choose degree with best (highest) score
    best_idx = np.argmax(mean_scores)
    best_degree = degrees[best_idx]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(range(maxdegree), error, "o-", color=BLUE, label="test error")
    ax.plot(range(maxdegree), bias, "s-", color=RED, label=r"bias$^2$ (+ $\sigma^2$)")
    ax.plot(range(maxdegree), variance, "d-", color=YELLOW, label="variance")
    # mark the optimal degree with a vertical line
    ax.axvline(x=best_degree, color="green", linestyle="--", label="optimal_degree")
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("MSE decomposition")
    ax.legend(frameon=False)


    plt.show()

    for d in (0, 4, 8, maxdegree - 1):
        print(f"degree {d:2d}: error {error[d]:.4f}  bias^2 {bias[d]:.4f}  "
            f"var {variance[d]:.4f}  sum {bias[d] + variance[d]:.4f}") 
    print(f"\noptimal degree: {best_degree}")

def bias_variance_sweep(n=40, noise=0.1, maxdegree=14, n_bootstraps=100,
                        make_model=None, seed=2026):
    mean_scores = []
    
    """Bootstrap estimate of error/bias^2/variance vs polynomial degree."""
    if make_model is None:
        make_model = lambda deg: make_pipeline(
            PolynomialFeatures(degree=deg), LinearRegression(fit_intercept=False))
    np.random.seed(seed)
    x = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x - 2)**2) + np.random.normal(0, noise, x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=seed)
    error, bias, variance = (np.zeros(maxdegree) for _ in range(3))
    for degree in range(maxdegree):
        model = make_model(degree)
        y_pred = np.empty((y_test.shape[0], n_bootstraps))
        for i in range(n_bootstraps):
            x_, y_ = resample(x_train, y_train)
            y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()
        error[degree] = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
        bias[degree] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
        variance[degree] = np.mean(np.var(y_pred, axis=1, keepdims=True))
        # negative MSE, so we will invert sign later
        scores = cross_val_score(
            model, x_train, y_train,
            scoring="neg_mean_squared_error",
            cv=10
        )
        
        mean_scores.append(scores.mean())
    
    mean_scores = np.array(mean_scores)

    # choose degree with best (highest) score
    best_idx = np.argmax(mean_scores)
    best_degree = degrees[best_idx]
    return error, bias, variance, best_degree