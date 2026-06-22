import io
import msoffcrypto
import pandas as pd

class Act_data_import:
    def __init__(self, input_dataset_file_path = 'excel_files/ACT_data_for_Ruogu_04AUG23.xlsx',
                 passw = 'password'):
        """
        self.input_dataset is meant to store the dataframe 
        and self.dict some metadata or configuration
        """
        self.input_dataset, self.dict = self.read_excel( input_dataset_file_path, passw)
        #df = self.input_dataset
            
    def read_excel(self, file_path, password=None):
        
        file = self.decrypt_file(file_path, password)

        df = pd.read_excel(file, sheet_name='data')
        dict = pd.read_excel(file, sheet_name='datadictionary')

        return df, dict
    def decrypt_file(self, file_path, password):
        """
        Decrypts an encrypted Excel file using the provided password.
        """
        with open(file_path, 'rb') as f:
            file = msoffcrypto.OfficeFile(f)
            if password:
                file.load_key(password=password)
            decrypted_file = io.BytesIO()
            file.decrypt(decrypted_file)
            decrypted_file.seek(0)
            return decrypted_file
    
if __name__ == "__main__":
    
    input_dataset_file_path = f'C:\\Users\\16473\\UFL Dropbox\\Junfu Cheng\\ActProject - Junfu Cheng\\Data\\act_clinico_demo_data_labeled_by_junfu_tara\\ACT_data_for_Ruogu_04AUG23.xlsx'
    passw = 'password'
    dataset = Act_data_import(input_dataset_file_path, passw)
    print("Dataset loaded successfully.")
    print(dataset.input_dataset.head())
    print(dataset.dict.head())
    print("Dataset shape:", dataset.input_dataset.shape)