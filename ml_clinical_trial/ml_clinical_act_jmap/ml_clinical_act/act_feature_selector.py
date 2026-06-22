from .ml_clinical.feature_selector import FeatureSelector
from .ml_clinical.data_importer import DataImporter

class ActFeatureSelector(FeatureSelector):
    def __init__(self, importer: DataImporter, verbose: bool = False):
        super().__init__(importer, verbose)

    def select_features(self, visit_times: list) -> list:
        """
        Select numeric features corresponding to specific visit times, excluding known non-informative features.
        Parameters:
        - visit_times (list of str): List of visit labels to filter features by (e.g., ['Visit 1', 'Visit 2'])
        Returns:
        - list: List of selected feature names
        """
        selected_features = []
        for feature in self.importer.get_feature_names():
            try:
                visit_meta = self.importer.get_feature_metadata(feature, 'Visit')
                type_meta = self.importer.get_feature_metadata(feature, 'type')
                if visit_meta in visit_times and type_meta == 'num':
                    selected_features.append(feature)
            except Exception:
                # Skip features missing metadata or not in dictionary
                continue

        return selected_features
    
    def select_feature_by_new_category_from_features(self, features: list, new_categories: list) -> list:
        """
        Select features based on a specific 'New Category' from the feature dictionary.
        Parameters:
        - features (list): List of feature names to filter.
        - new_category (str): The 'New Category' to filter features by.
        Returns:
        - list: List of features that match the specified 'New Category'.
        """
        selected_features = []
        for feature in features:
            try:
                category_meta = self.importer.get_feature_metadata(feature, 'New Category (2025_7_9)')
                if category_meta in new_categories:
                    selected_features.append(feature)
            except Exception:
                # Skip features missing metadata or not in dictionary
                continue
        return selected_features
    
    def rm_nan_features(self, subject_list:list, features: list) -> list:
        """
        Remove features that have NaN values across all subject IDs.
        Parameters:
        - features (list): List of feature names to filter.
        Returns:
        - list: List of features that do not have NaN values across all subject IDs.
        """
        valid_features = []
        for feature in features:
            try:
                # Check if the feature has any NaN values across all subject IDs
                data = self.importer.get_filtered_data(subject_list, [feature])
                if not data[feature].isnull().all():
                    valid_features.append(feature)
            except Exception:
                # Skip features that cannot be processed or do not exist
                continue

        return valid_features
