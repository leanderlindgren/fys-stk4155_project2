import numpy as np
from functions.linear_regression import Ridge_closed_form
from models.ols import OLS
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

class Ridge(OLS):
    def __init__(
            self,
            lmbd: float,
            degree: int,
            scale_mean: bool = True,
            scale_sd: bool = True
        ):
        self.scaler = StandardScaler(with_mean=scale_mean, with_std=scale_sd)
        self.lmbd = lmbd
        self.degree = degree

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray
        ) -> "Ridge":
        self.pipeline = make_pipeline(
            PolynomialFeatures(self.degree, include_bias=False),
            self.scaler
        )
        X_scaled = self.pipeline.fit_transform(x.reshape(-1, 1))
        self.intercept_ = y.mean()
        y_centered = y - y.mean()
        self.coef_ = Ridge_closed_form(X_scaled, y_centered, self.lmbd).ravel()
        return self