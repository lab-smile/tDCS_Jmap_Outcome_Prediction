from .ml_clinical_act.act_feature_type_identifier import ActFeatureTypeIdentifier
from .ml_clinical_act.act_importer_ml import ActDataImporterML
import pandas as pd

class JmapActFeatureTypeIdentifier(ActFeatureTypeIdentifier):
    def __init__(self, importer: ActDataImporterML, verbose: bool = False):
        super().__init__(importer,
                         verbose = verbose)
    
    def get_feature_type_lists(self, valid_features: list) -> tuple:
        """
        Split features into numerical and categorical based on 'Value' metadata.

        Parameters:
        - valid_features (list): List of feature names to be classified.
        Returns:
        - tuple: (numerical_features, categorical_features)
        """
        numerical_features = []
        categorical_features = []
        jmap_features = []

        for feature in valid_features:
            try:
                description = self.importer.get_feature_metadata(feature, 'varFormat')
                if pd.isna(description):
                    numerical_features.append(feature)
                elif description == "jmap":
                    jmap_features.append(feature)
                else:
                    categorical_features.append(feature)
            except Exception:
                # Skip features with missing metadata or raise a warning if needed
                continue
        
        print("numerical_features:", numerical_features)
        print("categorical_features:", categorical_features)
        print("jmap_features:", jmap_features)

        return numerical_features, categorical_features, jmap_features