import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import OneHotEncoder

def get_uniques_and_nan_count(df, column):
    print(column)
    print(f'Uniques: {df[column].unique()}')
    print(f'Number of Unique Values: {len(df[column].unique())}')
    print(f'Number of NaNs: {df[column].isna().sum()}')

def count_num_of_each_unique(df,column):
    print(df[column].value_counts())

def map_values(df, col):
    d = {v:k for k,v in dict(enumerate(df[col].unique())).items()}
    df[col] = df[col].map(d)

def encode(df, col):
    encoder = OneHotEncoder(handle_unknown='ignore')
    return encoder.fit_transform(df[col].values.reshape(-1,1)).toarray()

def dummy_and_concat(df, col, prefix):
    if isinstance(col, list) and isinstance(prefix, list):
        df_new = df
        for idx, column in enumerate(col):
            dummy_df = pd.get_dummies(df_new[col[idx]], prefix=prefix[idx], drop_first=True)
            df_new = pd.concat([df_new, dummy_df], axis=1)
        df_new.drop(col, axis=1, inplace=True)
        return df_new
    else:
        dummy_df = pd.get_dummies(df[col], prefix=prefix, drop_first=True) #prefix=prefix
        df = pd.concat([df, dummy_df], axis=1)
        df.drop(col, axis=1, inplace=True)
        return df



if __name__ == '__main__':
    # Read in excel file
    df = pd.read_excel('../../navigant_data/DemoData.xlsx')
    df.rename(columns={'AWO $ Bucket':'AWO_Bucket'}, inplace=True)

    #Iteravely make necessary dummies
    dummy_ls = ['LocationID', 'Service_Code', 'Procedure_Description', 'Financial_Class', 'NCI_Transaction_Detail', 'Admit_Type', \
                'Current_Patient_Class_Description', 'Discharge_Department_ID', 'Primary_Service_Name', 'Region']

    df_dummies = dummy_and_concat(df, dummy_ls, dummy_ls)

    # drop all dates here
    df.drop(['Service_Date', 'Post_Date', 'Account_Discharge_Date', 'Account_Billed_Date',
    'Admit_Date', 'First_Claim_Date', 'Last_Claim_Date', 'Last_Payment_Date'], axis=1, inplace=True)

    # drop specific features here #'Unnamed: 0'
    df.drop(['account_id', 'Transaction_Number', 'TransactionType_Tdata',
     'Procedure_Code', 'Insurance_Code', 'Insurance_Code_Description',
     'NCI_Transaction_Category', 'NCI_Transaction_Type', 'Account_Balance',
     'Admit_Diagnosis', 'Billing_Status', 'Medicaid_Pending_Flag', 'Medical_Record_Number',
     'Patient_Insurance_Balance', 'Patient_Self_Pay_Balance', 'Primary_Diagnosis',
     'ICD-10_Diagnosis_Principal', 'ICD-10_Diagnosis'], axis=1, inplace=True)

    # Create 6 different AWO categories - one for each AWO bucket
    df_dummies['AWO_Bucket'] = df['AWO_Bucket'].map({'1. <0': 0, '2. 0-1,000': 1,
    '3. 1,000-2,500':2,  '4. 2,500-5,000':3, '5. 5,000-10,000':4, '6. 10,000+':5})

    # df.to_csv('../data/trial_df_1.csv')
