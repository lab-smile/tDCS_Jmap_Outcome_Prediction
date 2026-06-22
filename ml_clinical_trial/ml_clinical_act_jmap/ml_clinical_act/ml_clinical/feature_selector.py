from .data_importer import DataImporter

class FeatureSelector:
    def __init__(self, importer: DataImporter, verbose: bool = False):
        """
        Initialize the feature selector with a DataImporter instance and a relative path.
        """
        self.importer = importer
        self.verbose = verbose


    def select_features(self, visit_times: list) -> list:
        """
        Select numeric features corresponding to specific visit times.

        Parameters:
        - visit_times (list of str): List of visit labels to filter features by (e.g., ['Visit 1', 'Visit 2'])

        Returns:
        - list: List of selected feature names
        """
        selected_features = []
        for feature in self.importer.get_feature_names():
            try:
                visit_meta = self.importer.get_feature_metadata(feature, 'Visit ')
                type_meta = self.importer.get_feature_metadata(feature, 'Type')
                if visit_meta in visit_times and type_meta == 'Num':
                    selected_features.append(feature)
            except Exception:
                # Skip features missing metadata or not in dictionary
                continue

        return selected_features
