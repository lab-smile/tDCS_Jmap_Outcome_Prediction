from .data_importer import DataImporter

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataSpiltter:
    def __init__(self, importer: DataImporter, responder_df: pd.DataFrame, relative_path: str, experiment_dir: str, random_state: int = 42):
        """
        Initialize the data splitter with a DataImporter instance and a relative path.
        """
        self.importer = importer
        self.responder_df = responder_df
        self.relative_path = relative_path
        self.experiment_dir = experiment_dir
        self.random_state = random_state

        #data = self.importer.load_data(filename='proact_final_feb2025.csv')
        subject_ids_origin = self.importer.get_subject_ids()
        subject_ids_responder_df = self.responder_df.index.tolist()
        #if data is None or data.empty:
        #    raise ValueError("Data not loaded or is empty. Please check the file path and content.")
        if not subject_ids_responder_df:
            raise ValueError("Subject IDs list is empty. Please ensure that the data has been cleaned and subject IDs are available.")
        
        # Ensure the responder_df has the all subject IDs included in the data
        if not set(subject_ids_responder_df).issubset(set(subject_ids_origin)):
            raise ValueError("Origin data does not contain all subject IDs from the responder DataFrame.")

    def split_data(self, train_ratio: float = 0.8):
        """
        Split the dataset into training and testing sets based on the specified ratio.

        Parameters:
        - train_ratio (float): The proportion of the dataset to include in the training set.

        Returns:
        - Tuple containing training and testing DataFrames.
        """
        # Example: train_ratio train, (1-train_ratio) test
        train_df, test_df = train_test_split(self.responder_df,
                                              test_size=(1-train_ratio),
                                              stratify=self.responder_df['responder'],  # stratification by target
                                                random_state=self.random_state)
        return train_df, test_df
        
    def split_data_by_site(self):
        """
        Split the dataset into training and testing sets based on the 'site' value.

        - Records with site == 1 or 2 go into the training set.
        - Records with site == 3 go into the testing set.

        Returns:
        - Tuple containing training and testing DataFrames (indexed by subject_id).
        """
        train_indices = []
        test_indices = []

        dictionary = self.importer.feature_dict
        dict_columns = dictionary.columns
        if 'site' not in dict_columns:
            for idx, row in self.responder_df.iterrows():
                subject_id = idx
                # The 'site' value is represented by the first character of the subject_id
                # '1' and '2' are from UF, '3' is from UA
                reverse = False
                if reverse == False:
                    site = int(str(subject_id)[0])
                    if site == 1 or site == 2:
                        train_indices.append(idx)
                    elif site == 3:
                        test_indices.append(idx)
                if reverse == True:
                    site = int(str(subject_id)[0])
                    if site == 3:
                        train_indices.append(idx)
                    elif site == 1 or site == 2:
                        test_indices.append(idx)
        if 'site' in dict_columns:
            for idx, row in self.responder_df.iterrows():
                subject_id = idx
                site = self.importer.get_value(subject_id, 'site')
                if site == 1:
                    train_indices.append(idx)
                elif site == 2:
                    test_indices.append(idx)

        train_df = self.responder_df.loc[train_indices]
        test_df = self.responder_df.loc[test_indices]
        return train_df, test_df


    
    def visualize_distribution(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Visualize and save the number of responders and non-responders in the training and testing sets
        as bar and pie charts with clear labels and number annotations.
        """

        experiment_dir = self.experiment_dir
        # Prepare the counts
        train_counts = train_df['responder'].value_counts().sort_index()
        test_counts = test_df['responder'].value_counts().sort_index()

        categories = ['Non-Responder', 'Responder']
        train_data = pd.DataFrame({'Category': categories, 'Count': train_counts.values})
        test_data = pd.DataFrame({'Category': categories, 'Count': test_counts.values})

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Responder Distribution in Training and Testing Sets', fontsize=16, weight='bold')

        # Bar chart - Train
        sns.barplot(data=train_data,
                        x='Category',
                        y='Count',
                        hue='Category',
                        ax=axes[0, 0], 
                        palette='Blues_d', 
                        legend=False)
        axes[0, 0].set_title('Training Set - Bar Chart')
        for i, count in enumerate(train_counts.values):
            axes[0, 0].text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')

        # Pie chart - Train
        axes[0, 1].pie(train_counts.values,
                    labels=categories,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=sns.color_palette('Blues'),
                    wedgeprops={'edgecolor': 'black'})
        axes[0, 1].set_title('Training Set - Pie Chart')

        # Bar chart - Test
        sns.barplot(data=test_data, 
                    x='Category', 
                    y='Count', 
                    hue='Category', 
                    ax=axes[1, 0], 
                    palette='Greens_d', 
                    legend=False)

        axes[1, 0].set_title('Testing Set - Bar Chart')
        for i, count in enumerate(test_counts.values):
            axes[1, 0].text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')

        # Pie chart - Test
        axes[1, 1].pie(test_counts.values,
                    labels=categories,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=sns.color_palette('Greens'),
                    wedgeprops={'edgecolor': 'black'})
        axes[1, 1].set_title('Testing Set - Pie Chart')

        # Save figure
        dataSpiltter_dir = os.path.join(experiment_dir, 'figure','DataSpiltter')
        dataSpiltter_path = os.path.join(dataSpiltter_dir, 'DataSpiltter_results.png')
        os.makedirs(dataSpiltter_dir, exist_ok=True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(dataSpiltter_path, dpi=600)
        # plt.show()
        plt.close()
        print(f"[DataSpiltter] Figure saved at: {dataSpiltter_path}")


    
# if __name__ == "__main__":
#     relative_path = 'data'
#     importer = DataImporter(relative_path)
#     cleaner = DataCleaner(importer, relative_path, versbose=False)
#     subject_ids = cleaner.get_cleaned_subject_ids()
    
#     labeler = DataLabeler(importer, subject_ids, relative_path)
#     responder_df_origin = labeler.label_data(label_columns=['total_womac', 'total_womac_v7'],
#                                        responder_criteria='above_median_decrease')
#     responder_df = responder_df_origin.dropna(subset=['responder'])
#     splitter = DataSpiltter(importer, responder_df, relative_path)
#     try:
#         train_df, test_df = splitter.split_data(train_ratio=0.8)
#         print("Data split completed successfully.")
#         print(f"Training set shape: {train_df.shape}")
#         print(f"The head of training set:\n{train_df.head()}")
#         print(f"The number of responder in training set: {train_df['responder'].value_counts().get(1, 0)}")
#         print(f"The number of non-responder in training set: {train_df['responder'].value_counts().get(0, 0)}")
        
#         print(f"Testing set shape: {test_df.shape}")
#         print(f"The head of testing set:\n{test_df.head()}")
#         print(f"The number of responder in testing set: {test_df['responder'].value_counts().get(1, 0)}")
#         print(f"The number of non-responder in testing set: {test_df['responder'].value_counts().get(0, 0)}")
#         splitter.visualize_distribution(train_df, test_df)
#     except Exception as e:
#         print(f"An error occurred during data labeling: {e}")
#         print("An error occurred during data labeling:")
#         traceback.print_exc()  # This prints the full traceback to stderr
        

        
