import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class AddMagnitudeFeature(BaseEstimator):
    def __init__(self, taxel_id):
        self.taxel_id = taxel_id

    def fit(self, X, y=None):
        return self  # No fitting necessary for this transformer

    def transform(self, X):
        X = X.copy()
        X[f'|XYZ|{self.taxel_id}'] = np.sqrt(
            X[f'X{self.taxel_id}'] ** 2 + X[f'Y{self.taxel_id}'] ** 2 + X[f'Z{self.taxel_id}'] ** 2)
        X[f'XZ{self.taxel_id}'] = X[f'X{self.taxel_id}'] * X[f'Z{self.taxel_id}']
        X[f'YZ{self.taxel_id}'] = X[f'Y{self.taxel_id}'] * X[f'Z{self.taxel_id}']
        # X = X.drop(columns=[f'X{self.taxel_id}', f'Y{self.taxel_id}'])
        return X


class FeaturePrinter(BaseEstimator):
    def __init__(self, prefix):
        self.prefix = prefix
    def fit(self, X, y=None):
        return self  # No fitting necessary for this transformer
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            print("FeaturePrinter: " + self.prefix + f'{list(X)}')
        else:
            print("FeaturePrinter: " + self.prefix + f'{len(list(X[0]))} features')
        return X


class SplitModel(BaseEstimator):
    def __init__(self, xy_pipeline, z_pipeline):
        self.xy_pipeline = xy_pipeline
        self.z_pipeline = z_pipeline
        self.steps = (self.xy_pipeline.steps, self.z_pipeline.steps)

    def fit(self, X, y=None):
        self.xy_pipeline.fit(X, y)
        self.z_pipeline.fit(X, y)
        self.steps = (self.xy_pipeline.steps, self.z_pipeline.steps)
        return self

    def transform(self, X):
        raise NotImplementedError("transform() not implemented for SplitModel")

    def predict(self, X):
        xy_pred = self.xy_pipeline.predict(X)
        z_pred = self.z_pipeline.predict(X)
        xyz_pred = xy_pred
        xyz_pred[:, 2] = np.array(z_pred)[:, 2]
        return xyz_pred

    def score(self, X, y):
        return (self.xy_pipeline.score(X, y), self.z_pipeline.score(X, y))
