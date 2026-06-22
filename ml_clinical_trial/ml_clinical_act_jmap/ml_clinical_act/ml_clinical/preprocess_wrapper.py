from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import numpy as np
import pandas as pd

class PreprocessWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, features, numerical_features, categorical_features, verbose=False):
        self.features = features
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.verbose = verbose

    def fit(self, X, y=None):
        # Only triggers fit_scaler_encoder=True once
        self.preprocess_data(
            X.copy(), 
            self.features, 
            self.numerical_features, 
            self.categorical_features, 
            fit_scaler_encoder=True, 
            verbose=self.verbose
        )
        self.used_numerical_features_ = self.numerical_features
        self.used_categorical_features_ = self.categorical_features
        return self

    def transform(self, X):
        return self.preprocess_data(
            X.copy(), 
            self.features, 
            self.numerical_features, 
            self.categorical_features, 
            fit_scaler_encoder=False, 
            verbose=self.verbose
        )
    
    def preprocess_data(self, data: pd.DataFrame, features, numerical_features, categorical_features, fit_scaler_encoder=False, verbose = True) -> pd.DataFrame:
        data = data.copy()
        data[numerical_features] = data[numerical_features].apply(pd.to_numeric, errors='coerce')
        data[numerical_features] = data[numerical_features].fillna(data[numerical_features].mean())
        
        nan_cols = data[numerical_features].columns[data[numerical_features].isna().all()]
        if len(nan_cols) > 0:
            if verbose:
                print("Dropping columns with all NaNs:", list(nan_cols))
            data.drop(columns=nan_cols, inplace=True)
            numerical_features = [col for col in numerical_features if col not in nan_cols]

        data[categorical_features] = data[categorical_features].astype(str).fillna('missing')

        if fit_scaler_encoder:
            self.scaler = StandardScaler()
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_num = pd.DataFrame(self.scaler.fit_transform(data[numerical_features]), columns=numerical_features, index=data.index)
            X_cat = pd.DataFrame(self.encoder.fit_transform(data[categorical_features]), index=data.index, columns=self.encoder.get_feature_names_out(categorical_features))
            self.used_numerical_features = numerical_features
            self.used_categorical_features = categorical_features
        else:
            # Ensure all training-time numerical features exist in the test data
            missing_numerical_cols = [col for col in self.used_numerical_features if col not in data.columns]
            if missing_numerical_cols:
                missing_df = pd.DataFrame(np.nan, index=data.index, columns=missing_numerical_cols)
                data = pd.concat([data, missing_df], axis=1)

            # Fill NaNs with training column means (safe because StandardScaler doesn't like NaNs)
            data[self.used_numerical_features] = data[self.used_numerical_features].fillna(0)

            X_num = pd.DataFrame(
                self.scaler.transform(data[self.used_numerical_features]),
                columns=self.used_numerical_features,
                index=data.index
            )

            # Same for categorical features
            missing_categorical_cols = [col for col in self.used_categorical_features if col not in data.columns]
            if missing_categorical_cols:
                missing_cat_df = pd.DataFrame('missing', index=data.index, columns=missing_categorical_cols)
                data = pd.concat([data, missing_cat_df], axis=1)


            data[self.used_categorical_features] = data[self.used_categorical_features].astype(str).fillna('missing')

            X_cat = pd.DataFrame(
                self.encoder.transform(data[self.used_categorical_features]),
                index=data.index,
                columns=self.encoder.get_feature_names_out(self.used_categorical_features)
            )

        return pd.concat([X_num, X_cat], axis=1)
