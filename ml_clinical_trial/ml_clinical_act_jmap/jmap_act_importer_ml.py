from .ml_clinical_act.act_importer_ml import ActDataImporterML

import nibabel as nib
import pandas as pd
import numpy as np
import os

class JmapActDataImporterML(ActDataImporterML):
    def __init__(self, 
                 relative_path, 
                 verbose = False,
                 printer = print):
        super().__init__(relative_path=relative_path, 
                         verbose=verbose)
        self.printer = printer
        self.relative_path

    def load_data(self, filename='act_data_generated.csv'):
        file_path = os.path.join(self.relative_path, filename)

        print(file_path)

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
        
        
    ###############
    # retrive data
    ##############
    def get_value(self, subject_id, feature_name):
        # Check that the dataset has been loaded before trying to use it
        if self.data is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")
        # Check that the requested feature (column) actually exists in the dataset
        if feature_name not in self.data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in the dataset.")
        # Check that the requested subject ID exists in the dataset's index (rows)
        if subject_id not in self.data.index:
            raise ValueError(f"Subject ID '{subject_id}' not found in the dataset.")
            
            
        if feature_name == 'jmap_tp1':
            value = self.get_jmap_value_at_tp1(subject_id)
        else:
            value = self.data.at[subject_id, feature_name]

        return value
    
    def get_jmap_value_at_tp1(self, subject_id):
        feature_name = 'jmap_tp1'
        if self.data is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")
        if feature_name not in self.data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in the dataset.")
        if subject_id not in self.data.index:
            raise ValueError(f"Subject ID '{subject_id}' not found in the dataset.")
            
        # Load volume
        path = self.data.at[subject_id, feature_name]
        if pd.isna(path) == True:
            volume = np.nan
        else:
            volume = nib.load(path).get_fdata().astype(np.float32)
            # Determine slice index
            #shape = volume.shape
            #print(shape)
            print("[JmapActDataImporterML] volume path:", path)
            print("[JmapActDataImporterML] volume shape:", volume.shape)

        return volume
    
    def get_filtered_data(self, subject_ids: list, selected_features: list) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Data not loaded yet. Call `load_data()` first.")

        if self.verbose:
            print(f"Headers of the dataset: {self.data.columns.tolist()}")
            print(f"Head of the dataset:\n{self.data.head()}")

        missing_features = set(selected_features) - set(self.data.columns)
        if missing_features:
            raise ValueError(f"Missing feature(s) in dataset: {missing_features}")


        filtered = self.data.loc[self.data.index.isin(subject_ids), selected_features]
        if 'jmap_tp1' in selected_features:
            filtered.loc[:, 'jmap_tp1'] = [self.get_jmap_value_at_tp1(sid) for sid in filtered.index]
            
        return filtered