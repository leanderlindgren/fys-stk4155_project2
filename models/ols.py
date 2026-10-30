import numpy as np
from types import NoneType
from functions.linear_regression import OLS_closed_form
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin

class OLS(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            degree: int,
            scale_mean: bool = True,
            scale_sd: bool = True
        ):
        self.scaler = StandardScaler(with_mean=scale_mean, with_std=scale_sd)
        self.degree = degree

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray
        ) -> "OLS":
        self.pipeline = make_pipeline(
            PolynomialFeatures(self.degree, include_bias=False),
            self.scaler
        )
        X_scaled = self.pipeline.fit_transform(x.reshape(-1, 1))
        self.intercept_ = y.mean()
        y_centered = y - y.mean()
        self.coef_ = OLS_closed_form(X_scaled, y_centered)
        return self

    def predict(
            self,
            x: np.ndarray
        ) -> np.ndarray:
        X_scaled = self.pipeline.transform(x.reshape(-1, 1))
        return X_scaled@self.coef_ + self.intercept_

    def mse(
            self,
            x: np.ndarray,
            y: np.ndarray
        ) -> float:
        pred = self.predict(x)
        return float(np.mean((pred - y)**2))