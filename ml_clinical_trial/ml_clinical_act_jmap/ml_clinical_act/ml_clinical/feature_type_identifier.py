from .data_importer import DataImporter


import pandas as pd

class FeatureTypeIdentifier:
    def __init__(self, importer: DataImporter, verbose: bool = False):
        """
        Initialize the feature selector with a DataImporter instance and a relative path.
        """
        self.importer = importer
        self.verbose = verbose
        #self.importer = DataImporter(relative_path)
        #cleaner = DataCleaner(self.importer, relative_path, versbose=False)
        #subject_ids = cleaner.get_cleaned_subject_ids()
        
        #labeler = DataLabeler(self.importer, subject_ids, relative_path)
        #responder_df_origin = labeler.label_data(label_columns=['total_womac', 'total_womac_v7'],
        #                                responder_criteria='above_median_decrease')
        #responder_df = responder_df_origin.dropna(subset=['responder'])
        #splitter = DataSpiltter(self.importer, responder_df, relative_path)
    
    def get_feature_type_lists(self, valid_features: list):
        """
        Split features into numerical and categorical based on 'Value' metadata.

        Parameters:
        - valid_features (list): List of feature names to be classified.

        Returns:
        - tuple: (numerical_features, categorical_features)
        """
        numerical_features = []
        categorical_features = []

        for feature in valid_features:
            if feature == 'age':
                numerical_features.append(feature)
                continue

            try:
                description = self.importer.get_feature_metadata(feature, 'Value')
                if pd.isna(description):
                    numerical_features.append(feature)
                else:
                    categorical_features.append(feature)
            except Exception:
                # Skip features with missing metadata or raise a warning if needed
                continue

        return numerical_features, categorical_features
