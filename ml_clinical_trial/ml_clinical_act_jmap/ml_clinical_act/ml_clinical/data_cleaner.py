from .data_importer import DataImporter
from .data_labeler import DataLabeler

import pandas as pd

class DataCleaner:
    def __init__(self, importer: DataImporter, relative_path: str = 'data', versbose: bool = False, printer = print):
        """
        Initialize the data cleaner with a DataImporter instance and a relative path.
        """
        self.importer = importer
        self.relative_path = relative_path
        self.versbose = versbose
        self.printer = printer

    def get_cleaned_subject_ids(self):
        subject_ids = self.importer.get_subject_ids()
        if self.versbose:
            self.printer("Data loaded successfully.")
            self.printer("unique subjects in the dataset:")
            for i, subject_id in enumerate(self.importer.get_subject_ids()):
                if i%15 == 14:
                    self.printer(subject_id)
                else:
                    self.printer(subject_id, end=', ')
            self.printer("")
        return subject_ids
    
    def get_cleaned_subject_ids_by_labeler(self):
        subject_ids = self.importer.get_subject_ids()
        labeler = DataLabeler(self.importer, subject_ids, self.relative_path)
        responder_df_origin = labeler.label_data(label_columns=['total_womac', 'total_womac_v7'],
                                            responder_criteria='above_median_decrease')
        responder_df = responder_df_origin.dropna(subset=['responder'])
        index_list = responder_df.index.tolist()
        return index_list
    
    def get_cleaned_subject_ids_by_selected_labeler(self, label_columns: list, responder_criteria: str ):
        subject_ids = self.importer.get_subject_ids()
        self.printer(f"Number of subject ids before cleaning by selected labeler: {len(subject_ids)}")
        labeler = DataLabeler(self.importer, subject_ids, self.relative_path)
        responder_df_origin = labeler.label_data(label_columns=label_columns,
                                            responder_criteria=responder_criteria)
        responder_df = responder_df_origin.dropna(subset=['responder'])
        index_list = responder_df.index.tolist()
        self.printer(f"Number of cleaned subject ids by selected labeler: {len(index_list)}")
        return index_list
    
    def clean_data_by_site_and_group(self, index_list: list):
        """
        Clean the dataset using the given index list by:
        1. Dropping entries with missing 'site' values.
        2. Dropping participants who do not have any entry with group_value == 4.

        Parameters:
        - index_list (list): List of row indices to include in the cleaning process.

        Returns:
        - cleaned_index_list (list): Filtered list of indices after applying the cleaning rules.
        """
        cleaned_indices = []
        subject_ids_with_site = {}
        subject_ids_with_group = {}

        # First, filter out indices where 'site' is None or NaN
        print("Cleaning data by site and group...")
        print(f"Total indices before cleaning: {len(index_list)}")
        for idx in index_list:
            subject_id = idx
            site = self.importer.get_value(subject_id, 'site')

            if pd.notna(site):  # only keep rows with non-null site values
                cleaned_indices.append(idx)
                if subject_id not in subject_ids_with_site:
                    subject_ids_with_site[subject_id] = []
                subject_ids_with_site[subject_id].append(site)

        print(f"Total indices after removing missing 'site': {len(cleaned_indices)}")

        cleaned_indices = []
        for idx in subject_ids_with_site.keys():
            subject_id = idx
            site = self.importer.get_value(subject_id, 'Group')

            if pd.notna(site):  # only keep rows with non-null site values
                cleaned_indices.append(idx)
                if subject_id not in subject_ids_with_group:
                    subject_ids_with_group[subject_id] = []
                subject_ids_with_group[subject_id].append(site)
        print(f"Total indices after removing missing 'group': {len(cleaned_indices)}")

        # Then, keep only those subject_ids that have at least one group == 4
        final_indices = []
        for idx in cleaned_indices:
            subject_id = idx
            groups = subject_ids_with_group.get(subject_id, [])
            if 4 in groups:
                final_indices.append(idx)

        print(f"Total indices after keeping only those with group == 4: {len(final_indices)}")

        return final_indices
    
    def clean_data_by_site_and_group(self, index_list: list, site_var_name: str, group_var_name: str, group_value: list):
        """
        Clean the dataset using the given index list by:
        1. Dropping entries with missing 'site' values.
        2. Dropping participants who do not have any entry with group == group_value.

        Parameters:
        - index_list (list): List of row indices to include in the cleaning process.

        Returns:
        - cleaned_index_list (list): Filtered list of indices after applying the cleaning rules.
        """
        cleaned_indices = []
        subject_ids_with_site = {}
        subject_ids_with_group = {}

        # First, filter out indices where 'site' is None or NaN
        print("Cleaning data by site and group...")
        print(f"Total indices before cleaning: {len(index_list)}")
        for idx in index_list:
            subject_id = idx
            site = self.importer.get_value(subject_id, site_var_name)

            if pd.notna(site):  # only keep rows with non-null site values
                cleaned_indices.append(idx)
                if subject_id not in subject_ids_with_site:
                    subject_ids_with_site[subject_id] = []
                subject_ids_with_site[subject_id].append(site)

        print(f"Total indices after removing missing {site_var_name}: {len(cleaned_indices)}")

        cleaned_indices = []
        for idx in subject_ids_with_site.keys():
            subject_id = idx
            group = self.importer.get_value(subject_id, group_var_name)

            if pd.notna(group):  # only keep rows with non-null group values
                cleaned_indices.append(idx)
                if subject_id not in subject_ids_with_group:
                    subject_ids_with_group[subject_id] = []
                subject_ids_with_group[subject_id].append(group)
        print(f"Total indices after removing missing {group_var_name}: {len(cleaned_indices)}")

        # Then, keep only those subject_ids that have at least one site == group_value
        final_indices = []
        for idx in cleaned_indices:
            subject_id = idx
            groups = subject_ids_with_group.get(subject_id, [])
            for site in group_value:
                if site in groups:
                    final_indices.append(idx)
                    break

        print(f"Total indices after keeping only those with {group_var_name} == {group_value}: {len(final_indices)}")

        return final_indices
    
    

    
# if __name__ == "__main__":
#     relative_path = 'data'
#     importer = DataImporter(relative_path)
#     cleaner = DataCleaner(importer, relative_path, versbose=True)
    
#     try:
#         subject_ids = cleaner.get_cleaned_subject_ids()
#         print("Data cleaning completed successfully.")
#         print(cleaner.get_cleaned_subject_ids_by_labeler())
#     except Exception as e:
#         print(f"An error occurred during data cleaning: {e}")