import os
import pandas as pd

class ActDataProcessor:
    def __init__(self, dataset, output_dir='../data_generation_log/act_data'):
        # original clinico-demographic data from Dr. Yunfeng Dai
        self.df = dataset.input_dataset
        self.dict = dataset.dict
        
        # new formated data based on original data from Dr. Yunfeng Dai 
        self.generated_data = None
        self.generated_dict = None
        
        # output directory with new formated data saved
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def check_columns(self):
        # Check required columns
        # check if 'tp' or 'subjectid' exist in table
        if 'tp' not in self.df.columns or 'subjectid' not in self.df.columns:
            raise ValueError("Input dataset must contain 'tp' and 'subjectid' columns.")
        # Validate 'tp' column
        valid_tp_values = {0, 1, 2, 3, 4}
        # check if invalid 'tp' value exits
        invalid_tp_rows = self.df[~self.df['tp'].isin(valid_tp_values)]
        if not invalid_tp_rows.empty:
            raise ValueError(
                f"Invalid 'tp' values found in the dataset:\n{invalid_tp_rows[['subjectid', 'tp']].drop_duplicates()}"
            )
    
    def pivot_dataframe(self):
        # Pivot the dataframe to wide format
        grouped = self.df.set_index(['subjectid', 'tp'])
        wide_df = grouped.unstack(level='tp')
        wide_df.columns = [f"{col[0]}_tp{int(col[1])}" for col in wide_df.columns]
        wide_df.reset_index(inplace=True)
        return wide_df
    
    def generate_dict_from_original(self):
        # Build the updated dictionary with Visit column
        var_dict = []
        for _, row in self.dict.iterrows():
            varName = row['varName']
            if varName in ['subjectid', 'tp']:
                continue  # Skip base identifiers

            for tp in sorted(self.df['tp'].dropna().unique()):
                new_row = row.copy()
                new_row['varName'] = f"{varName}_tp{int(tp)}"
                new_row = pd.concat([pd.Series({'Visit': int(tp)}), new_row])
                var_dict.append(new_row)

        # Add static subjectid entry (non-timepointed)
        subjectid_row = self.dict[self.dict['varName'] == 'subjectid']
        if not subjectid_row.empty:
            row = subjectid_row.iloc[0].copy()
            row = pd.concat([pd.Series({'Visit': 'static'}), row])
            var_dict.append(row)
        # Combine and reorder columns
        generated_dict = pd.DataFrame(var_dict)
        cols = ['Visit'] + [col for col in generated_dict.columns if col != 'Visit']
        generated_dict = generated_dict[cols]
        return generated_dict

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

    def save(self):
        if self.generated_data is not None:
            self.generated_data.to_csv(os.path.join(self.output_dir, 'act_data_generated.csv'), index=False)
            print(f"Generated data saved to {os.path.abspath(self.output_dir)}")
        if self.generated_dict is not None:
            self.generated_dict.to_csv(os.path.join(self.output_dir, 'act_data_dict_generated.csv'), index=False)
            print(f"Generated dictionary saved to {os.path.abspath(self.output_dir)}")

if __name__ == "__main__":
    from act_importer import Act_data_import  # adjust if necessary

    input_dataset_file_path = r'/home/junfu.cheng/SMILE/TARA/data_2025_8_10/ACT_data_for_Ruogu_04AUG23.xlsx'
    passw = 'password'

    # Step 1: Load dataset
    dataset = Act_data_import(input_dataset_file_path, passw)
    print("Dataset loaded successfully.")
    print(dataset.input_dataset.head())
    print(dataset.dict.head())
    print("Dataset shape:", dataset.input_dataset.shape)

    # Step 2: Process data
    processor = ActDataProcessor(dataset)
    processor.process()
    processor.save()

    
