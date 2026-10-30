import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression

def OLS_closed_form(
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
    return np.linalg.lstsq(X, y)[0]

def descent_OLS(X,y,theta_target, gamma, maxs, n, n_features, num_iters=1000):
    distances_OLS = []
    theta_gdOLS = np.zeros(n_features)

    # Gradient descent loops
    for t in range(num_iters):
        grad_OLS = 2/n * X.T @ (X @ theta_gdOLS - y)
        theta_gdOLS -= gamma * grad_OLS
        dist = np.linalg.norm(theta_gdOLS - theta_target)
        distances_OLS.append(dist)

        # Stopping criterion on gradient norm
        if np.linalg.norm(grad_OLS) < maxs:
            print(f"gamma={gamma}: converged after {t+1} iterations for OLS, dist={dist:.2e}")
            print("Gradient Descent OLS coefficients:", theta_gdOLS)
            break
    return distances_OLS

# def descent_Ridge(X,y,theta_target, gamma, maxs, n, n_features, num_iters=1000):
#     theta_gdRidge = np.zeros(n_features)
#     distances_Ridge = []

#     for t in range(num_iters):
#         # Compute gradients for Ridge
#         grad_Ridge = 2/n * X.T @ (X @ theta_gdRidge - y) + 2 * lam * theta_gdRidge #lambda = 0.01
#         theta_gdRidge -= gamma * grad_Ridge
#         dist = np.linalg.norm(theta_gdRidge - theta_target)
#         distances_Ridge.append(dist)

#         # Stopping criterion on gradient norm
#         if np.linalg.norm(grad_Ridge) < maxs:
#             print(f"gamma={gamma}: converged after {t+1} iterations for Ridge")
#             print("Gradient Descent Ridge coefficients:", theta_gdRidge)
#             break

#     return distances_Ridge

def optimiser_step(method, theta, g, state, t, gamma, beta=0.9, rho=0.99,
                   beta1=0.9, beta2=0.999, eps=1e-8):
    """One update of theta from the gradient g at step t (t = 1, 2, ...), Eqs. (4.10), (4.28),
    (4.42)-(4.45), (4.47)-(4.48) and (4.51)-(4.55).  state carries the running quantities."""
    if method == "plain":
        return theta - gamma * g, state
    if method == "momentum":
        v = beta * state.get("v", 0.0) + gamma * g               # Eq. (4.28)
        state["v"] = v
        return theta - v, state
    if method == "adagrad":
        r = state.get("r", 0.0) + g * g                           # Eq. (4.42)
        state["r"] = r
        return theta - gamma * g / (np.sqrt(r) + eps), state      # Eq. (4.45)
    if method == "rmsprop":
        r = rho * state.get("r", 0.0) + (1.0 - rho) * g * g       # Eq. (4.47)
        state["r"] = r
        return theta - gamma * g / (np.sqrt(r) + eps), state      # Eq. (4.48)
    if method == "adam":
        m = beta1 * state.get("m", 0.0) + (1.0 - beta1) * g       # Eq. (4.51)
        r = beta2 * state.get("r", 0.0) + (1.0 - beta2) * g * g   # Eq. (4.52)
        state["m"], state["r"] = m, r
        m_hat = m / (1.0 - beta1**t)                              # Eq. (4.54)
        r_hat = r / (1.0 - beta2**t)
        return theta - gamma * m_hat / (np.sqrt(r_hat) + eps), state   # Eq. (4.55)
    raise ValueError(f"unknown method {method}")

def optimise(grad, theta0, method, gamma, num_iters=1000, tol=0.0, **kw):
    """Run one optimiser from theta0 with the full gradient; returns all iterates."""
    theta, state = np.array(theta0, dtype=float), {}
    history = [theta.copy()]
    for t in range(1, num_iters + 1):
        g = grad(theta)
        theta, state = optimiser_step(method, theta, g, state, t, gamma, **kw)
        history.append(theta.copy())
        if np.linalg.norm(g) < tol:
            break
    return np.array(history)

# #Might need to look at this. I was getting weird inpredictable behavior out of this in project 1.
# #Sometimes it showed a correct Lasso curve. Sometimes it showed clean straight lines?
# def lasso(y):
#     lasso_est = np.sign(y) * np.maximum(np.abs(y) - lam/2., 0.)
#     print(lasso_est)

#     fig, ax = plt.subplots(figsize=(6.0, 4.2))
#     ax.plot(y, lasso_est, color="#BB5566", lw=2,
#             label=r"Lasso: $S_{\lambda/2}(y_i)$ (3.63)")
#     ax.axvspan(-lmbda/2, lmbda/2, color="#BB5566", alpha=0.08)
#     ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
#     ax.set_xlabel(r"data value $y_i$"); ax.set_ylabel(r"$\hat\theta_i$")
#     ax.set_title(r"Identity design, $\lambda=1$", fontsize=10)
#     ax.legend(fontsize=9, loc="upper left"); ax.set_aspect("equal")
#     plt.tight_layout(); plt.show()