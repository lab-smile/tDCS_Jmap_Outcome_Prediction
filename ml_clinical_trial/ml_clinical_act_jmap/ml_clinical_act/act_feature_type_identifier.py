from .ml_clinical.feature_type_identifier import FeatureTypeIdentifier
from .ml_clinical.data_importer import DataImporter

import pandas as pd

class ActFeatureTypeIdentifier(FeatureTypeIdentifier):
    def __init__(self, importer: DataImporter, verbose: bool = False):
        """
        Initialize the feature selector with a DataImporter instance and a relative path.
        """
        super().__init__(importer, verbose = verbose)

    def get_feature_type_lists(self, valid_features):
        """
        Split features into numerical and categorical based on 'Value' metadata.

        Parameters:
        - valid_features (list): List of feature names to be classified.
        Returns:
        - tuple: (numerical_features, categorical_features)
        """
        numerical_features = []
        categorical_features = []

        print('hello')

        for feature in valid_features:
            try:
                description = self.importer.get_feature_metadata(feature, 'varFormat')
                if pd.isna(description):
                    numerical_features.append(feature)
                else:
                    categorical_features.append(feature)
            except Exception:
                # Skip features with missing metadata or raise a warning if needed
                continue

        
        print('numerical_features:', numerical_features)
        print('categorical_features:', categorical_features)

        return numerical_features, categorical_features
