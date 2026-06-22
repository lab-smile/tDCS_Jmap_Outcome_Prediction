import pandas as pd
from pathlib import Path
import os
from .ml_clinical.data_importer import DataImporter

class ActDataImporterML(DataImporter):
    def __init__(self, relative_path='../data_generation_log/act_data', verbose=False, printer = print):
        # Normalize relative_path in case it's a tuple
        relative_path = self._normalize_paths(relative_path)
        # Convert to absolute path relative to current file
        script_dir = Path(__file__).resolve().parent
        abs_path = (script_dir / relative_path).resolve()

        super().__init__(relative_path=str(abs_path), verbose=verbose)
        self.id_column = 'subjectid'
        self.script_dir = script_dir
        self.relative_path = str(abs_path)
        self.data = None
        self.feature_dict = None
        self.printer = printer

    ###############
    # load data
    ##############
    def load_data(self, filename='act_data_generated.csv'):
        # Normalize relative_path in case it's a tuple
        filename = self._normalize_paths(filename)

        file_path = os.path.join(self.relative_path, filename)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file '{filename}' was not found at '{file_path}'.")


        df = pd.read_csv(file_path)

        if self.id_column not in df.columns:
            raise ValueError(f"The dataset does not contain a '{self.id_column}' column.")

        df.set_index(self.id_column, inplace=True)
        self.data = df

        if self.verbose:
            self.printer(f"Data loaded successfully from: {file_path}")
            self.printer(f"Shape: {self.data.shape}")
        return self.data

    def load_feature_dictionary(self, dict_filename='act_data_dict_generated.csv'):
        # Normalize dict_filename in case it's a tuple
        dict_filename = self._normalize_paths(dict_filename)
        
        file_path = os.path.join(self.relative_path, dict_filename)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file '{dict_filename}' was not found at '{file_path}'.")

        self.feature_dict = pd.read_csv(file_path)
        if self.verbose:
            self.printer(f"Feature dictionary loaded successfully from: {file_path}")
        return self.feature_dict

    ###############
    # retrive data
    ##############
    def get_subject_ids(self):
        if self.data is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")
        return self.data.index.unique().tolist()

    def get_value(self, subject_id, feature_name):
        if self.data is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")
        if feature_name not in self.data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in the dataset.")
        if subject_id not in self.data.index:
            raise ValueError(f"Subject ID '{subject_id}' not found in the dataset.")

        return self.data.at[subject_id, feature_name]

    def get_feature_column(self, feature_name):
        if self.data is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")
        if feature_name not in self.data.columns:
            raise KeyError(f"Feature '{feature_name}' not found in dataset.")
        return self.data[[feature_name]]

    def get_filtered_data(self, subject_ids: list, selected_features: list) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")

        if self.verbose:
            self.printer(f"Headers of the dataset: {self.data.columns.tolist()}")
            self.printer(f"Head of the dataset:\n{self.data.head()}")

        missing_features = set(selected_features) - set(self.data.columns)
        if missing_features:
            raise ValueError(f"Missing feature(s) in dataset: {missing_features}")

        filtered = self.data.loc[self.data.index.isin(subject_ids), selected_features]
        return filtered

    ###############
    # check feature dictionary
    ##############
    def get_feature_metadata(self, feature_name: str, field: str) -> str:
        if self.feature_dict is None:
            raise ValueError("Feature dictionary not loaded yet. Call `load_feature_dictionary()` first.")
        if field not in self.feature_dict.columns:
            raise ValueError(f"Field '{field}' not found in the dictionary columns.")

        match = self.feature_dict[self.feature_dict['varName'] == feature_name]
        if match.empty:
            raise ValueError(f"Feature '{feature_name}' not found in the dictionary.")

        return match.iloc[0][field]
    
    def get_feature_description(self, feature_name: str):
        """
        Retrieve the description of a feature from the loaded dictionary.

        Parameters:
        - feature_name (str): The name of the feature (must match the 'Variable' column)

        Returns:
        - str: Description of the feature

        Raises:
        - ValueError if dictionary not loaded or feature not found
        """
        if self.feature_dict is None:
            raise ValueError("Feature dictionary not loaded yet. Call `load_feature_dictionary()` first.")

        match = self.feature_dict[self.feature_dict['varName'] == feature_name]
        if match.empty:
            raise ValueError(f"Feature '{feature_name}' not found in the dictionary.")

        return match.iloc[0]['varDesc']

    @staticmethod
    def _normalize_paths(filepath):
        # Normalize relative_path in case it's a tuple
        if isinstance(filepath, tuple):
            filepath = Path(*filepath)
        else:
            filepath = Path(filepath)
        return filepath

if __name__ == "__main__":

    print("Running full functional test for ActDataImporterML...")

    
    def export_redcap_visit_map(importer: DataImporter, output_csv='../../data_generation_log/act_data/'):
        if importer.feature_dict is None:
            raise ValueError("Feature dictionary not loaded. Please run load_feature_dictionary() first.")
        
        # Select only the 'REDCap Form' and 'Visit ' columns
        sub_df = importer.feature_dict[['New Category (2025_7_9)', 'Visit']].dropna()

        # Group by REDCap Form and collect unique visit values
        grouped = (
            sub_df.groupby('New Category (2025_7_9)')['Visit']
            .apply(lambda x: sorted(set(x.dropna())))
            .reset_index()
            .rename(columns={'Visit': 'Visit List'})
        )

        
        

        # Save to CSV
        # Ensure output directory exists
        os.makedirs(output_csv, exist_ok=True)
        grouped.to_csv(os.path.join(output_csv, 'ActNewCate.csv'), index=False)
        print(f"Saved ActNewCate Form to Visit mapping to {os.path.join(output_csv, 'ActNewCate.csv')}")

        return grouped

    try:
        output_csv = '../../data_generation_log/act_data/'
        relative_path = output_csv
        # Instantiate
        importer = ActDataImporterML(relative_path = relative_path, verbose=True)
        print("[PASS] Initialization OK")

        # Load dataset
        df = importer.load_data()
        print(f"[PASS] Data loaded. Shape: {df.shape}")

        # Load dictionary
        feature_dict = importer.load_feature_dictionary()
        print(f"[PASS] Dictionary loaded. Shape: {feature_dict.shape}")

        # Feature names
        feature_names = importer.get_feature_names()
        assert isinstance(feature_names, list) and len(feature_names) > 0
        print(f"[PASS] Feature names fetched: {feature_names[:5]}")

        # Subject IDs
        subject_ids = importer.get_subject_ids()
        assert isinstance(subject_ids, list) and len(subject_ids) > 0
        print(f"[PASS] Subject IDs fetched: {subject_ids[:5]}")

        # Sample value lookup
        test_id = subject_ids[0]
        test_feature = feature_names[1]
        value = importer.get_value(test_id, test_feature)
        print(f"[PASS] Value for ({test_id}, {test_feature}): {value}")

        # Feature column
        col_df = importer.get_feature_column(test_feature)
        print(f"[PASS] Feature column retrieved. Shape: {col_df.shape}")

        # Filtered subset
        filtered = importer.get_filtered_data(subject_ids[:5], feature_names[:3])
        print(f"[PASS] Filtered data retrieved. Shape: {filtered.shape}")

        # Metadata test (dictionary column safety)
        metadata_fields = [col for col in feature_dict.columns if col != 'varName']
        for field in metadata_fields[:3]:  # test 3 fields
            try:
                meta = importer.get_feature_metadata(test_feature, field)
                print(f"[PASS] Metadata '{field}' for '{test_feature}': {meta}")
            except Exception as e:
                print(f"[WARNING]  Skipped metadata field '{field}': {e}")

        # Description via parent method (if applicable)
        try:
            desc = importer.get_feature_description(test_feature)
            print(f"[PASS] Feature description for '{test_feature}': {desc}")
        except Exception as e:
            print(f"[WARNING] Feature description not available: {e}")

        # Optional: Export REDCap mapping (if the columns exist)
        try:
            mapping = export_redcap_visit_map(importer, output_csv = output_csv)
            print(f"[PASS] REDCap to Visit map saved. Preview:\n{mapping.head()}")
        except Exception as e:
            print(f"[WARNING]  REDCap-Visit map skipped: {e}")

        print("[FINISH] All functional checks passed.")

    except Exception as e:
        print("[ERROR] Functional test failed:", str(e))


