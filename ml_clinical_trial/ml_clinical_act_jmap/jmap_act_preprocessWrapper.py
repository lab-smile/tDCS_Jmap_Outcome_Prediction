from .ml_clinical_act.ml_clinical.preprocess_wrapper import PreprocessWrapper
from .jmap_preprocessor import JmapACTPreprocessor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import numpy as np
import pandas as pd

class JmapACTPreprocessWrapper(PreprocessWrapper):
    def __init__(self, 
                 features, 
                 numerical_features, 
                 categorical_features, 
                 jmap_features, 
                 verbose=False):
        super().__init__(features, 
                         numerical_features, 
                         categorical_features,  
                         verbose = verbose)
        self.jmap_features = jmap_features
        
    def fit(self, X, y=None):
        # Only triggers fit_scaler_encoder=True once
        self.preprocess_data(
            X.copy(), 
            self.features, 
            self.numerical_features, 
            self.categorical_features,
            self.jmap_features,
            fit_scaler_encoder=True, 
            verbose=self.verbose
        )
        self.used_numerical_features_ = self.numerical_features
        self.used_categorical_features_ = self.categorical_features
        self.used_jmap_features_ = self.jmap_features
        return self

    def transform(self, X):
        return self.preprocess_data(
            X.copy(), 
            self.features, 
            self.numerical_features, 
            self.categorical_features,
            self.jmap_features,
            fit_scaler_encoder=False, 
            verbose=self.verbose
        )
    
    def preprocess_data(self, 
                        data: pd.DataFrame, 
                        features, 
                        numerical_features, 
                        categorical_features, 
                        jmap_features,
                        fit_scaler_encoder=True, 
                        verbose = True) -> pd.DataFrame:
        data = data.copy()
        # Tries to convert every value to a number (float or int).
        # If a value can’t be converted (e.g., "abc" or "N/A"), 
        # it replaces it with NaN (Not a Number) because of errors='coerce'.
        data[numerical_features] = data[numerical_features].apply(pd.to_numeric, errors='coerce')
        
        # For each numerical column, calculates the mean of that column.
        # Replaces any missing values (NaN) with that column’s mean.
        data[numerical_features] = data[numerical_features].fillna(data[numerical_features].mean())
        
        
        # Checks data[numerical_features].isna().all() → returns True for any column where all values are NaN.
        # Uses .columns[...] to extract just the names of those "all-NaN" columns.
        nan_cols = data[numerical_features].columns[data[numerical_features].isna().all()]
        if len(nan_cols) > 0:
            if verbose:
                print("Dropping columns with all NaNs:", list(nan_cols))
            data.drop(columns=nan_cols, inplace=True)
            numerical_features = [col for col in numerical_features if col not in nan_cols]

        # TConverts every value in categories columns to a string type.
        # .fillna('missing')
        data[categorical_features] = data[categorical_features].astype(str).fillna('missing')

        if fit_scaler_encoder:
            self.scaler = StandardScaler()
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.jmap_preprocessor = JmapACTPreprocessor()

            # Numerical features
            X_num = pd.DataFrame(
                self.scaler.fit_transform(data[numerical_features]),
                columns=numerical_features,
                index=data.index
            )

            # Categorical features
            X_cat = pd.DataFrame(
                self.encoder.fit_transform(data[categorical_features]),
                index=data.index,
                columns=self.encoder.get_feature_names_out(categorical_features)
            )

            # JMAP features
            X_jmap_transformed = self.jmap_preprocessor.fit_transform(data[jmap_features])
            X_jmap = pd.DataFrame(
                X_jmap_transformed,
                index=data.index,
                columns=self.jmap_preprocessor.get_feature_names_out(jmap_features)
            )

            self.used_numerical_features = numerical_features
            self.used_categorical_features = categorical_features
            self.used_jmap_features = jmap_features
        else:
            # Ensure all training-time numerical features exist in the test data
            missing_numerical_cols = [col for col in self.used_numerical_features if col not in data.columns]
            if missing_numerical_cols:
                missing_df = pd.DataFrame(np.nan, index=data.index, columns=missing_numerical_cols)
                data = pd.concat([data, missing_df], axis=1)

            # Fill NaNs with training column means (safe because StandardScaler doesn't like NaNs)
            data[self.used_numerical_features] = data[self.used_numerical_features].fillna(
                data[self.used_numerical_features].mean()
            )

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
            
            # --- JMAP ---
            if getattr(self, "used_jmap_features_", None):
                # Align columns for jmap features too
                missing_jmap = [c for c in self.used_jmap_features_ if c not in data.columns]
                if missing_jmap:
                    data = pd.concat([data, pd.DataFrame(np.nan, index=data.index, columns=missing_jmap)], axis=1)

                X_jmap_arr = self.jmap_preprocessor.transform(data[self.used_jmap_features_])
                if hasattr(self.jmap_preprocessor, "get_feature_names_out"):
                    jmap_cols = self.jmap_preprocessor.get_feature_names_out(self.used_jmap_features_)
                else:
                    jmap_cols = self.used_jmap_features_
                X_jmap = pd.DataFrame(X_jmap_arr, index=data.index, columns=jmap_cols)
            else:
                X_jmap = pd.DataFrame(index=data.index)

        return pd.concat([X_num, X_cat, X_jmap], axis=1)
    
    