from .data_importer import DataImporter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataLabeler:
    def __init__(self, data_importer: DataImporter, subject_ids: list, relative_path: str = 'data', verbose: bool = False, experiment_dir: str = './results'):
        self.data_importer = data_importer
        self.relative_path = relative_path
        self.subject_ids = subject_ids
        self.verbose = verbose
        self.experiment_dir = experiment_dir

    def label_data(self, label_columns: list, responder_criteria: str = 'decrease') -> pd.DataFrame:
        """
        Load data and apply responder labeling based on the specified label column and criteria.

        Parameters:
            label_column (list): The list of the name of the column to use for labeling.
            responder_criteria (str): Criteria for defining responders. One of:
                ['decrease', 'increase', 'above_median_increase', 'above_median_decrease'].

        Returns:
            pd.DataFrame: DataFrame with ['subjectid', 'responder'].
        """
        
        
        # Label responders using the specified criteria and target column
        responder_df = self.label_responders(
            responder_criteria=responder_criteria,
            subjectid_list=self.subject_ids,
            targets=label_columns
        )


        return responder_df
        
    def label_responders(self, responder_criteria, subjectid_list, targets):
        """
        Label subjects as responders (1) or non-responders (0) based on change in target values 
        between timepoints TP1 and TP2.

        Parameters:
            responder_criteria (str): One of 'decrease', 'increase', 
                                      'above_median_increase', 'above_median_decrease'.
            subjectid_list (list): List of subject IDs to evaluate.
            targets (list): Target column names.

        Returns:
            pd.DataFrame: DataFrame with ['subjectid', 'responder'].
        """
        data = self.data_importer.data
        dictionary = self.data_importer.load_feature_dictionary()
        if data is None or data.empty:
            raise ValueError("Data not loaded or is empty. Please check the file path and content.")
        if dictionary is None or dictionary.empty:
            raise ValueError("Feature dictionary not loaded or is empty. Please check the file path and content.")
        
        feature_list_from_origin_data = self.data_importer.get_feature_names()
        subjectid_list_from_origin_data = self.data_importer.get_subject_ids()

        # Validate required columns
        required_columns = set(targets)
        if not required_columns.issubset(feature_list_from_origin_data):
            missing = required_columns - set(feature_list_from_origin_data)
            raise ValueError(f"Missing required columns in dataset: {missing}")
        
        # Check if targets are num type of data
        dictionary_columns = self.data_importer.feature_dict.columns
        if 'Type' not in dictionary_columns:
            for target in targets:
                if not self.data_importer.get_feature_metadata(target, 'type') == 'num':
                    raise ValueError(f"Target column '{target}' must be 'num' type.")
        if 'type' not in dictionary_columns:
            for target in targets:
                if not self.data_importer.get_feature_metadata(target, 'Type') == 'Num':
                    raise ValueError(f"Target column '{target}' must be 'Num' type.")

        # Filter for valid subjects
        df_target1 = self.data_importer.get_feature_column(targets[0])
        if df_target1 is None or df_target1.empty:  
            raise ValueError(f"Target column '{targets[0]}' not found in the dataset.") 
        if not set(subjectid_list).issubset(set(subjectid_list_from_origin_data)):
            missing_subjects = set(subjectid_list) - set(subjectid_list_from_origin_data)
            raise ValueError(f"Missing subject IDs in dataset: {missing_subjects}")
        
        df_target2 = self.data_importer.get_feature_column(targets[1])
        if df_target2 is None or df_target2.empty:      
            raise ValueError(f"Target column '{targets[1]}' not found in the dataset.")
        if not set(subjectid_list).issubset(set(subjectid_list_from_origin_data)):
            missing_subjects = set(subjectid_list) - set(subjectid_list_from_origin_data)
            raise ValueError(f"Missing subject IDs in dataset: {missing_subjects}")
        
        if df_target1.shape[0] != df_target2.shape[0]:
            print(f"Shape of df_target1: {df_target1.shape}")
            print(f"Shape of df_target2: {df_target2.shape}")
            print(f"Shape of subjectid_list: {len(subjectid_list)}")
            if len(df_target1.index) != len(df_target2.index):
                print(f"different subjectid_list: {set(subjectid_list) - set(df_target1.index)}")
                print(f"different subjectid_list: {set(subjectid_list) - set(df_target2.index)}")
            raise ValueError("Target columns must have the same number of rows.")

        df = df_target1.merge(df_target2, left_index=True, right_index=True)
        if self.verbose:
            print(f"labeler dataset shape: {df.shape}")
            print(f"Unique subjects in filtered dataset:")
            for i, subject_id in enumerate(df.index):
                if i%15 == 14:
                    print(subject_id)
                else:
                    print(subject_id, end=', ')
            print()
            print(f"df head:\n{df.head()}")

        results = []
        median_value_delta = 0
        above_median = 0
        below_median = 0
        equal_median = 0

        # === SWITCH 1: decrease ===
        if responder_criteria == 'decrease':
            for sid in subjectid_list:
                tp_pre = df.at[sid, targets[0] ]
                tp_post = df.at[sid, targets[1] ]
                if pd.isna(tp_pre) or pd.isna(tp_post):
                    responder = None
                else:
                    val1, val2 = tp_pre, tp_post
                    delta = val2 - val1
                    responder = int(delta < 0)
                results.append({'subject_id': sid, 'responder': responder})

        # === SWITCH 2: increase ===
        elif responder_criteria == 'increase':
            for sid in subjectid_list:
                tp_pre = df.at[sid, targets[0] ]
                tp_post = df.at[sid, targets[1] ]
                if pd.isna(tp_pre) or pd.isna(tp_post):
                    responder = None
                else:
                    val1, val2 = tp_pre, tp_post
                    delta = val2 - val1
                    responder = int(delta > 0)
                results.append({'subject_id': sid, 'responder': responder})

        # === SWITCH 3: above_median_increase ===
        elif responder_criteria == 'above_median_increase':
            deltas = []
            for sid in subjectid_list:
                tp_pre = df.at[sid, targets[0] ]
                tp_post = df.at[sid, targets[1] ]
                if not pd.isna(tp_pre) and not pd.isna(tp_post):
                    delta = tp_post - tp_pre
                    deltas.append(delta)
            print('len(deltas):', len(deltas))
            median_value = np.median(deltas)

            for sid in subjectid_list:
                tp_pre = df.at[sid, targets[0] ]
                tp_post = df.at[sid, targets[1] ]
                if pd.isna(tp_pre) or pd.isna(tp_post):
                    responder = None
                else:
                    val1, val2 = tp_pre, tp_post
                    delta = val2 - val1
                    if delta > median_value:
                        responder = 1
                        above_median += 1
                    elif delta < median_value:
                        responder = 0
                        below_median += 1
                    else:
                        responder = 0
                        equal_median += 1
                        median_value_delta += 1  # optional if you need this counter elsewhere
                results.append({'subject_id': sid, 'responder': responder})
            print(f"Subjects above median delta: {above_median}")
            print(f"Subjects below median delta: {below_median}")
            print(f"Subjects with delta equal to median: {equal_median}")

        # === SWITCH 4: above_median_decrease ===
        elif responder_criteria == 'above_median_decrease':
            deltas = []
            for sid in subjectid_list:
                tp_pre = df.at[sid, targets[0]]
                tp_post = df.at[sid, targets[1] ]
                if not pd.isna(tp_pre) and not pd.isna(tp_post):
                    delta = tp_pre - tp_post
                    deltas.append(delta)
            print('len(deltas):', len(deltas))
            median_value = np.median(deltas)

            for sid in subjectid_list:
                tp_pre = df.at[sid, targets[0] ]
                tp_post = df.at[sid, targets[1] ]
                if pd.isna(tp_pre) or pd.isna(tp_post):
                    responder = None
                else:
                    val1, val2 = tp_pre, tp_post
                    delta = val1 - val2
                    if delta > median_value:
                        responder = 1
                        above_median += 1
                    elif delta < median_value:
                        responder = 0
                        below_median += 1
                    else:
                        responder = 0
                        equal_median += 1
                        median_value_delta += 1  # optional if you need this counter elsewhere
                results.append({'subject_id': sid, 'responder': responder})
            print(f"Subjects above median delta: {above_median}")
            print(f"Subjects below median delta: {below_median}")
            print(f"Subjects with delta equal to median: {equal_median}")

        elif responder_criteria == 'clinical_decrease':
            for sid in subjectid_list:
                tp_pre = df.at[sid, targets[0]]
                tp_post = df.at[sid, targets[1]]
                
                if pd.isna(tp_pre) or pd.isna(tp_post):
                    responder = None
                else:
                    # Condition for clinical decrease
                    if tp_pre >= 39 and tp_post < 39:
                        responder = 1
                    else:
                        responder = 0
                        
                results.append({'subject_id': sid, 'responder': responder})
            
            # Optional: Count responders
            num_responders = sum(r['responder'] == 1 for r in results if r['responder'] is not None)
            print(f"Number of clinical decrease responders: {num_responders}")

        elif responder_criteria == 'above_median_decrease_in_severe':
            deltas = []
            # Calculate deltas only for severe subjects (tp_pre >= 39)
            ids_in_severe = []
            for sid in subjectid_list:
                tp_pre = df.at[sid, targets[0]]
                tp_post = df.at[sid, targets[1]]
                if not pd.isna(tp_pre) and not pd.isna(tp_post) and tp_pre >= 39:
                    delta = tp_pre - tp_post
                    deltas.append(delta)
                    ids_in_severe.append(sid)
            print(f"[DataLabler] Number of severe subjects (tp_pre >= 39, len(deltas) for severe subjects): {len(deltas)}")
            
            median_value = np.median(deltas) if deltas else None
            print("[DataLabler] Median Value: ", median_value)

            for sid in subjectid_list:
                tp_pre = df.at[sid, targets[0]]
                tp_post = df.at[sid, targets[1]]
                if pd.isna(tp_pre) or pd.isna(tp_post) or median_value is None:
                    responder = None
                else:
                    delta = tp_pre - tp_post
                    # Only classify responder if tp_pre is severe (>= 39), otherwise responder = 0
                    if tp_pre >= 39:
                        if delta > median_value:
                            responder = 1
                            above_median += 1
                        elif delta < median_value:
                            responder = 0
                            below_median += 1
                        else:
                            responder = 0
                            equal_median += 1
                            median_value_delta += 1  # optional counter
                    else:
                        responder = 0  # Not severe, so not responder
                results.append({'subject_id': sid, 'responder': responder})

            print(f"[DataLabeler] Severe subjects above median delta: {above_median}")
            print(f"[DataLabeler] Severe subjects below median delta: {below_median}")
            print(f"[DataLabeler] Severe subjects with delta equal to median: {equal_median}")


            plot_spaghetti = True            
            if plot_spaghetti:
                
                targets = ['stai_state_score_tp1', 'stai_state_score_tp2']
                self.variable_name='stai_state_score'
                # Convert to DataFrame
                results_plot = pd.DataFrame(results)
                
                # Filter rows where subject_id is in ids_in_severe
                results_plot = results_plot[results_plot['subject_id'].isin(ids_in_severe)].set_index('subject_id').sort_index()

                # calculate descriptive statistics for the target feature at each time point, stratified by responder status
                # and inferential statistics to compare the target feature between responders and non-responders at each time point
                from .spaghetti_plot import stats_aggregate, extract_score_data_from_ids, StatsTimePoint
                tp1_df, tp2_df, responder = extract_score_data_from_ids(df, ids_in_severe, results_plot, targets)
                stats_aggregate(tp1_df, tp2_df, responder, self, StatsTimePoint.PRE)
                print(f"[DataLabeler] stats_aggregate done before the data cleaning")

        else:
            raise ValueError(f"Invalid responder_criteria: {responder_criteria}")

        responder_df = pd.DataFrame(results).set_index('subject_id')

        if self.verbose:
            # Plot
            # Ensure output directory exists
            os.makedirs('./results/figure/dataLabeler', exist_ok=True)
            # Count responders
            responder_counts = responder_df['responder'].value_counts(dropna=True).sort_index()

            # Set seaborn style
            sns.set(style='whitegrid')

            # Define colors for both charts
            colors = ['#E74C3C', '#2ECC71']  # red (non-responder), green (responder)
            labels = ['Non-Responder (0)', 'Responder (1)']

            # Create figure with 1 row, 2 columns
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # --- Bar Chart ---
            bars = axes[0].bar(responder_counts.index.astype(str), responder_counts.values, color=colors)
            for bar in bars:
                yval = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.2, int(yval),
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

            axes[0].set_xticks([0, 1])
            axes[0].set_xticklabels(labels)
            axes[0].set_ylabel('Number of Subjects')
            axes[0].set_title('Responder vs Non-Responder Count')

            # --- Pie Chart ---
            axes[1].pie(responder_counts.values, labels=labels, autopct='%1.1f%%',
                        startangle=90, colors=colors, textprops={'fontsize': 10})
            axes[1].set_title('Responder Distribution (Pie Chart)')

            # Tight layout and save
            plt.tight_layout()
            plt.savefig('./results/figure/dataLabeler/labeler_results.png', dpi=300)
            plt.show()

            
        return responder_df
    
    def get_labeled_subject_ids(self, responder_df: pd.DataFrame) -> list:
        """
        Get the list of subject IDs that have been labeled as responders.

        Parameters:
            responder_df (pd.DataFrame): DataFrame containing subject IDs and their responder labels.

        Returns:
            list: List of subject IDs labeled as responders.
        """
        responder_list = responder_df[responder_df['responder'] == 1].index.tolist()
        non_responder_list = responder_df[responder_df['responder'] == 0].index.tolist()
        return responder_list + non_responder_list
    
# if __name__ == "__main__":
#     relative_path = 'data'
#     importer = DataImporter(relative_path)
#     cleaner = DataCleaner(importer, relative_path, versbose=False)
#     subject_ids = cleaner.get_cleaned_subject_ids()
    
#     labeler = DataLabeler(importer, subject_ids, relative_path)

    
#     try:
#         responder_df = labeler.label_data(label_columns=['total_womac', 'total_womac_v7'], responder_criteria='above_median_decrease')
#         print("Data labeling completed successfully.")
#         print(responder_df.head())
#         print("labeled subject IDs:" )
#         for i, subject_id in enumerate(labeler.get_labeled_subject_ids(responder_df)):
#             if i % 15 == 14:
#                 print(subject_id)
#             else:
#                 print(subject_id, end=', ')
#         print()
#         print("labeled subject IDs length:", len(labeler.get_labeled_subject_ids(responder_df)))
#         os.makedirs('./results/table/dataLabeler', exist_ok=True)
#         responder_df.to_csv('./results/table/dataLabeler/responder_labels.csv', index=True)


#     except Exception as e:
#         print(f"An error occurred during data labeling: {e}")
#         print("An error occurred during data labeling:")
#         traceback.print_exc()  # This prints the full traceback to stderr
        


