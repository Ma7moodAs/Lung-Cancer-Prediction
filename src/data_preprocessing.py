import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
#from sklearn.compose import ColumnTransformer
from pathlib import Path


# Dataset Loading
def load_dataset(file_path):
    """
    Dataset loading function.
    """
    try:
        df = pd.read_csv(file_path)
        print('Dataset Loaded Successfully')
        return df
    except Exception as error:
        print(f' Error loading dataset: {error}')
        return None

# Duplicated values check:
def check_duplicates(df):
    '''
    Duplicated values check function.
    '''
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        print(f'The DataFrame contains {duplicates_count} duplicated rows.')
        print(df[df.duplicated()])
    else:
        print('The dataframe contains no duplicated rows.')


# Duplicated Values Removal:
def remove_duplicates(df):
    """
    Remove duplicated values from the DataFrame.
    """
    initial_df_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape
    print(f'{initial_df_shape[0] - final_shape[0]} duplicated rows has been removed from the dataset')
    return df
# Numerical, and categorical columnns identification:
def indentify_col_type(df):
    """
    separate numerical and categorical columns in the DataFrame.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    return numerical_cols,categorical_cols

def save_preprocessed_data(df,file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(file_path,index=False)   


df = load_dataset(r'C:\Users\User\Desktop\Lung Cancer Prediction\data\survey lung cancer.csv')
check_duplicates(df)
df = remove_duplicates(df)
numerical_cols,categorical_cols = indentify_col_type(df)

# Preprocessing Dataframe:
for col in categorical_cols:
    if col != 'GENDER':
        df[col] = df[col].str.lower().map({'yes':1,'no':0})
    else:
        df[col] = df[col].str.lower().map({"male":1,"female":0})
df = df.drop(columns=['GENDER','SHORTNESS_OF_BREATH'],errors='ignore')
print('Data Preprocessing Completed')

# save preprocessed Dataframe
new_df = save_preprocessed_data(df,r'C:\Users\User\Desktop\Lung Cancer Prediction\data\preprocessed_lung_cancer_data.csv')
print('Preprocessed data saved successfully.')
