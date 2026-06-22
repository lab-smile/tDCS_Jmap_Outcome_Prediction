from .ml_clinical.data_cleaner import DataCleaner

import pandas as pd
import numpy as np

class ActDataCleaner(DataCleaner):
    def __init__(self, importer, relative_path, verbose=False):
        super().__init__(importer, relative_path, verbose)
        self.importer = importer
        self.relative_path = relative_path
        self.verbose = verbose
    
    def clean_data_by_ids(self, index_list, ids):
        cleaned_indices = []
        print("Cleaning data by ids...")
        print(f"Total indices before cleaning: {len(index_list)}")
        for idx in index_list:
            if idx not in ids:
                cleaned_indices.append(idx)
        print(f"Total indices after cleaning by ids: {len(cleaned_indices)}")
        return cleaned_indices

    def clean_data_by_site_and_group(self, index_list: list, group_var_name: str, group_value: list):
        """
        Clean the dataset using the given index list by:
        1. Dropping entries with missing 'site' values.
        2. Dropping participants who do not have any entry with site == 4.

        Parameters:
        - index_list (list): List of row indices (participant id) to include in the cleaning process.

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

            # The 'site' value is represented by the first character of the subject_id
            # '1' and '2' are from UF, '3' is from UA
            site = int(str(subject_id)[0])

            if pd.notna(site):  # only keep rows with non-null site values
                cleaned_indices.append(idx)
                if subject_id not in subject_ids_with_site:
                    subject_ids_with_site[subject_id] = []
                subject_ids_with_site[subject_id].append(site)

        print(f"Total indices after removing missing site: {len(cleaned_indices)}")

        cleaned_indices = []
        for idx in subject_ids_with_site.keys():
            subject_id = idx
            group = self.importer.get_value(subject_id, group_var_name)

            if pd.notna(group):  # only keep rows with non-null site values
                cleaned_indices.append(idx)
                if subject_id not in subject_ids_with_group:
                    subject_ids_with_group[subject_id] = []
                subject_ids_with_group[subject_id].append(group)
        print(f"Total indices after removing missing {group_var_name}: {len(cleaned_indices)}")

        # Then, keep only those subject_ids that have at least one group == group_value
        final_indices = []
        for idx in cleaned_indices:
            subject_id = idx
            groups = subject_ids_with_group.get(subject_id, [])
            for group in group_value:
                if group in groups:
                    final_indices.append(idx)
                    break

        print(f"Total indices after keeping only those with {group_var_name} == {group_value}: {len(final_indices)}")

        return final_indices

    def clean_data_by_feature_values(
        self, 
        index_list: list, 
        feature_name: str, 
        criteria_value: float, 
        criteria: str):
        """
        index_list is the list of participant id as dataframe is using participant ids as indices
        """

        cleaned_indices = []
        for idx in index_list:
            subject_id = idx
            feature_value = self.importer.get_value(subject_id, feature_name)
            if criteria == 'greater than or equal to':
                if feature_value >= criteria_value:
                    cleaned_indices.append(subject_id)
            if criteria == 'less than or equal to':
                if feature_value <= criteria_value:
                    cleaned_indices.append(subject_id)
            if criteria == 'equal to':
                if feature_value == criteria_value:
                    cleaned_indices.append(subject_id)
            if criteria == 'valid':
                # np.isnan(feature_value).all() checks if all elements in the array are NaN — if not, consider it valid.
                if isinstance(feature_value, np.ndarray):
                    # For arrays: valid if it contains at least one non-NaN value
                    if not np.isnan(feature_value).all():
                        cleaned_indices.append(subject_id)
                else:
                    # For scalars: valid if not NaN
                    if pd.notna(feature_value):
                        cleaned_indices.append(subject_id)

        return cleaned_indices
    