import pandas as pd
import warnings
import os

class DataImporter:
    def __init__(self, relative_path: str, verbose: bool = False, printer = print):
        """
        Initialize the data importer with a relative file path to the dataset folder.

        Parameters:
        - relative_path (str): Path to the folder containing the CSV and XLSX files, relative to the script location.
        """
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.relative_path = relative_path
        self.data = None
        self.feature_dict = None
        self.verbose = verbose
        self.printer = printer

    

    def load_data(self, filename='proact_final_feb2025.csv'):
        """
        Load the CSV data into a pandas DataFrame.

        Parameters:
        - filename (str): Name of the CSV file (default: 'proact_final_feb2025.csv')

        Returns:
        - pd.DataFrame: Loaded DataFrame
        """
        file_path = os.path.join(self.script_dir, self.relative_path, filename)
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file '{filename}' was not found at '{file_path}'.")

        # pandas is reading your CSV file, 
        # some columns (listed in the warning) contain mixed data types 
        # (e.g., both numbers and strings)
        # suppress the DtypeWarning 
        # for just the pd.read_csv line using the warnings context manager. 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.DtypeWarning)
            self.data = pd.read_csv(file_path)
        
        if self.verbose:
            self.printer(f"Data loaded successfully from: {file_path}")
            self.printer(f"Shape: {self.data.shape}")
        return self.data

    def load_feature_dictionary(self, dict_filename='PROACT Complete Data Dictionary.xlsx', sheet_name='Complete Data Dictionary'):
        """
        Load the feature dictionary from an Excel file using proper column names.

        Parameters:
        - dict_filename (str): Excel file name (default: 'PROACT Complete Data Dictionary.xlsx')
        - sheet_name (str): Sheet name containing the dictionary

        Returns:
        - pd.DataFrame: Loaded feature dictionary
        """
        file_path = os.path.join(self.script_dir, self.relative_path, dict_filename)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file '{dict_filename}' was not found at '{file_path}'.")

        self.feature_dict = pd.read_excel(file_path, sheet_name=sheet_name)
        expected_cols = ['Variable', 'Description']
        if not all(col in self.feature_dict.columns for col in expected_cols):
            raise ValueError(f"The sheet must contain columns: {expected_cols}")

        if self.verbose:
            self.printer(f"Feature dictionary loaded successfully from: {file_path}")
        return self.feature_dict
    
    def get_feature_names(self):
        """
        Retrieve the names of the features (columns) in the loaded dataset.

        Returns:
        - list: List of feature names
        """
        if self.data is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")
        
        return self.data.columns.tolist()

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

        match = self.feature_dict[self.feature_dict['Variable'] == feature_name]
        if match.empty:
            raise ValueError(f"Feature '{feature_name}' not found in the dictionary.")

        return match.iloc[0]['Description']


    def print_feature_names(self):
        """
        Print the feature (column) names from the loaded dataset.
        """
        if self.data is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")
        
        self.printer("Feature (column) names:")
        for i, col in enumerate(self.data.columns):
            self.printer(f"{i+1}. {col}")

    def get_subject_ids(self):
        """
        Retrieve unique subject IDs from the dataset.

        Returns:
        - list: Unique subject IDs
        """
        if self.data is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")

        if 'subject_id' not in self.data.columns:
            raise ValueError("The dataset does not contain a 'subject_id' column.")

        return self.data['subject_id'].unique().tolist()

    def get_value(self, subject_id, feature_name):
        """
        Retrieve the value for a given subject ID and feature name.

        Parameters:
        - subject_id (str or int): The subject identifier
        - feature_name (str): The column/feature name

        Returns:
        - The value at the intersection of subject and feature
        """
        if self.data is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")

        if 'subject_id' not in self.data.columns:
            raise ValueError("The dataset does not contain a 'subject_id' column.")

        if feature_name not in self.data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in the dataset.")

        row = self.data[self.data['subject_id'] == subject_id]
        if row.empty:
            raise ValueError(f"Subject ID '{subject_id}' not found in the dataset.")

        return row.iloc[0][feature_name]
    
    def get_feature_metadata(self, feature_name: str, field: str) -> str:
        """
        Retrieve a specific metadata field for a given feature.

        Parameters:
        - feature_name (str): The name of the feature (must match the 'Variable' column)
        - field (str): The metadata field to retrieve. Must be one of:
          'Type', 'Len', 'Format', 'Informat', 'REDCap Form', 'Visit', 'Description', 'Value'

        Returns:
        - str: The value of the requested metadata field

        Raises:
        - ValueError if dictionary not loaded, feature not found, or field is invalid
        """
        allowed_fields = [
            'Type', 'Len', 'Format', 'Informat',
            'REDCap Form', 'Visit ', 'Description', 'Value'
        ]

        if self.feature_dict is None:
            raise ValueError("Feature dictionary not loaded yet. Call `load_feature_dictionary()` first.")

        if field not in allowed_fields:
            raise ValueError(f"Invalid field '{field}'. Must be one of: {allowed_fields}")

        match = self.feature_dict[self.feature_dict['Variable'] == feature_name]
        if match.empty:
            raise ValueError(f"Feature '{feature_name}' not found in the dictionary.")

        return match.iloc[0][field]

    def get_feature_column(self, feature_name):
        """
        Returns a DataFrame with 'subject_id' as index and the specified feature as the only column.

        Parameters:
        - feature_name (str): Name of the column to extract.

        Returns:
        - pandas.DataFrame: DataFrame with 'subject_id' as index and the feature column.

        Raises:
        - KeyError: If 'subject_id' or the feature column is not found in self.data.
        """
        required_columns = {'subject_id', feature_name}
        missing = required_columns - set(self.data.columns)
        if missing:
            raise KeyError(f"Missing required column(s): {missing}")

        return self.data.set_index('subject_id')[[feature_name]]
    
    def get_filtered_data(self, subject_ids: list, selected_features: list) -> pd.DataFrame:
        """
        Return a DataFrame filtered by selected subject IDs and feature columns.

        Parameters:
        - selected_features (list): List of feature (column) names to include.
        - subject_ids (list): List of subject IDs to filter.
suppress the DtypeWarning for just the pd.read_csv line using the warnings context manager. 
        Returns:
        - pd.DataFrame: DataFrame with subject_ids as index and selected_features as columns.

        Raises:
        - ValueError: If data is not loaded or if required columns are missing.
        """
        if self.data is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")

        missing_features = set(selected_features) - set(self.data.columns)
        if 'subject_id' not in self.data.columns:
            raise ValueError("The dataset does not contain a 'subject_id' column.")
        if missing_features:
            raise ValueError(f"Missing feature(s) in dataset: {missing_features}")

        filtered = self.data[self.data['subject_id'].isin(subject_ids)][['subject_id'] + selected_features]
        return filtered.set_index('subject_id')

if __name__ == "__main__":
    from logger import Logger

    # Example usage:
    log_path = f'../../data_generation_log/test_logs/data_importer'
    log_filename = f'test.log'
    logger = Logger(log_path, 
                log_filename)

    relative_path = '../../data_generation_log/act_data'
    filename = 'act_data_generated.csv'
    verbose = True

    printer = logger.write_log



    try:
        # Instantiate
        importer = DataImporter(relative_path = relative_path,
                            verbose = verbose,
                            printer = printer) 
        logger.write_log("[PASS] Initialization OK")

        # Load dataset
        df = importer.load_data(filename = filename)
        print(f"[PASS] Data loaded. Shape: {df.shape}")
    
    except Exception as e:
        logger.write_log("[ERROR] Functional test failed:", str(e))




# if __name__ == "__main__":
#     # Example usage:
#     relative_path = './data'
#     importer = DataImporter(relative_path)
#     df = importer.load_data()
#     # importer.print_feature_names()
#     value = importer.get_value(1002, 'age')
#     print(f"Value for 1002, age: {value}")

#     # Print a column
#     age_column = importer.get_feature_column('age')
#     print("Age column data:")
#     print(age_column.head())

#     # Load the feature dictionary
#     importer.load_feature_dictionary()

#     # Get specific metadata for 'age'
#     feature_type = importer.get_feature_metadata('age', 'Type')
#     description = importer.get_feature_metadata('systolic_bp', 'Visit ')

#     print(f"Type of 'systolic_bp': {feature_type}")
#     print(f"Value of 'systolic_bp': {description}")

#     def export_redcap_visit_map(importer: DataImporter, output_csv='./results/table/dataImporter/redcap_form_to_visits.csv'):
#         if importer.feature_dict is None:
#             raise ValueError("Feature dictionary not loaded. Please run load_feature_dictionary() first.")
        
#         # Select only the 'REDCap Form' and 'Visit ' columns
#         sub_df = importer.feature_dict[['REDCap Form', 'Visit ']].dropna()

#         # Group by REDCap Form and collect unique visit values
#         grouped = (
#             sub_df.groupby('REDCap Form')['Visit ']
#             .apply(lambda x: sorted(set(x.dropna())))
#             .reset_index()
#             .rename(columns={'Visit ': 'Visit List'})
#         )

#         # Save to CSV
#         # Ensure output directory exists
#         os.makedirs('./results/table/dataImporter', exist_ok=True)
#         grouped.to_csv(output_csv, index=False)
#         print(f"Saved REDCap Form to Visit mapping to {output_csv}")

#         return grouped

#     # Usage example:
#     mapping_df = export_redcap_visit_map(importer)
