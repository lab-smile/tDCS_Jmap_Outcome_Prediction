from .ml_clinical_act.act_data_processor import ActDataProcessor

import pandas as pd
import numpy as np
import os
import re

class JmapActDataProcessor(ActDataProcessor):
    def __init__(self, dataset, output_dir='../data_generation_log/act_data'):
        super().__init__(dataset, output_dir)
        self.root_dir = "/blue/camctrp/working/junfu.cheng/jmap_dataset_for_autoencoder"
        #self.root_dir = "/orange/ruogu.fang/data/ACT/act_jmap_in_mni_by_spm12_junfu_cheng"
        self.file_filter = 'wT1_tDCSLAB_Jbrain.nii'
        
    def process(self):
        
        # Check 'tp' or 'subjectid' columns
        self.check_columns()

        # Drop duplicates just in case
        # It removes completely duplicate rows from the dataset self.df. 
        self.df = self.df.drop_duplicates()

        # Pivot the dataframe to wide format
        self.generated_data = self.pivot_dataframe()

        # Build the updated dictionary with Visit columnl()
        self.generated_dict = self.generate_dict_from_original()
        
        # New jmap row to add to dict
        new_row = {
            "Visit": 1,
            "varnum": 1256,
            "varName": "jmap_tp1",
            "varDesc": "jmap",
            "type": "num",
            "length": float("nan"),
            "varFormat": "jmap",
            "info": "J-map registrated by SPM12 (v6906) with ICBM space template - European brains option",
            "Original Category": float("nan"),
            "New Category (2025_7_9)": "jmap",
            "Comment": ""
        }
        # Append the row
        self.generated_dict = pd.concat([self.generated_dict, pd.DataFrame([new_row])], ignore_index=True)
        

        # get all sub-directories in root directories
        all_subdirs = self.get_numeric_name_directories(self.root_dir)
        # get the participant_id-file_path map based on name of directories
        subject_file_map = self.get_subjectid_based_on_directories(all_subdirs, self.file_filter)
        
        # Add column to dataframe
        # Take the subjectid column from the dataframe self.generated_data (which is a Series of IDs, like 100031, 100161, ...)
        # For each value in that column, look it up as a key in the dictionary subject_file_map.
        # Replace the value with whatever the dictionary has for that key.
        # If self.generated_data doesn’t have exactly the same set of subjectids as the folder list, then
        # .map(subject_file_map) will still work fine, but
        # Any subjectid in the dataframe not in subject_file_map will get NaN 
        # (which is also stored internally as a NumPy NaN (np.nan called float("nan") from NumPy).).
        # (they behave identically for operations like .isna() or np.isnan().)
        # Any folder subjectid not in the dataframe is simply ignored.
        self.generated_data["jmap_tp1"] = self.generated_data["subjectid"].map(subject_file_map)
        
        # mapped_paths is a Series
        # It contains either a file path string or NaN for each row.
        mapped_paths = self.generated_data["subjectid"].map(subject_file_map)
        # Debug: find which subjects in the dataframe had no matching folder
        # .isna() returns a Boolean mask the same length as mapped_paths, with True where the value is NaN and False otherwise.
        missing_subjects = self.generated_data.loc[mapped_paths.isna(), "subjectid"]
        if not missing_subjects.empty:
            print("No file found for subjects:", missing_subjects.tolist())
            
        # Debug: subjects with a folder but not in dataframe
        folder_subjects = set(subject_file_map.keys())
        df_subjects = set(self.generated_data["subjectid"])
        unmatched_folders = folder_subjects - df_subjects
        if unmatched_folders:
            print("Folders found but not in dataframe:", sorted(unmatched_folders))
        
        
    def get_numeric_name_directories(self, root_dir):
        # Get only numeric directories (subjectid)
        all_subdirs = [
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and re.match(r'^\d+$', d)
        ]
        return all_subdirs
    
    def get_subjectid_based_on_directories(self, all_subdirs, file_filter):
        # Make a lookup: subjectid → filepath if file exists, else 0
        subject_file_map = {}
        for subdir in sorted(all_subdirs):
            subject_id = int(os.path.basename(subdir))
            target_file = os.path.join(subdir, file_filter)
            if os.path.isfile(target_file):
                subject_file_map[subject_id] = target_file
            else:
                subject_file_map[subject_id] = np.nan
                
        return subject_file_map
                
        
if __name__ == "__main__":
    from .ml_clinical_act.act_importer import Act_data_import  # adjust if necessary

    input_dataset_file_path = r'/home/junfu.cheng/SMILE/TARA/data_2025_8_10/ACT_data_for_Ruogu_04AUG23.xlsx'
    passw = 'password'
    output_dir = './data_generation_log/act_data'

    # Step 1: Load dataset
    dataset = Act_data_import(input_dataset_file_path, passw)
    print("Dataset loaded successfully.")
    print(dataset.input_dataset.head())
    print(dataset.dict.head())
    print("Dataset shape:", dataset.input_dataset.shape)

    # Step 2: Process data
    processor = JmapActDataProcessor(dataset, output_dir = output_dir)
    processor.process()
    processor.save()