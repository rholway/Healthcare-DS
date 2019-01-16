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

def dummy_and_concat(df, col):
    dummy_df = pd.get_dummies(df[col], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop([col], axis=1, inplace=True)
    return df



if __name__ == '__main__':
    # Read in excel file
    df = pd.read_excel('../../../data/DemoData.xlsx')
    df.rename(columns={'AWO $ Bucket':'AWO_Bucket'}, inplace=True)

    # Take sample of df for easier, quicker manipulation to start
    # df = df.sample(n=1000)

    # Save sample df
    # df.to_csv('../data/sample_df')

    # Use sample df
    # df = pd.read_csv('../data/sample_df')
    # df.rename(columns={'AWO $ Bucket':'AWO_Bucket'}, inplace=True)

    # Create Dummy Variables and Drop the following
    # location_id_dummied_df = pd.get_dummies(df['LocationID'], drop_first=True)
    # df = pd.concat([df, location_id_dummied_df], axis=1)
    # df.drop('LocationID', axis=1, inplace=True)
    #
    # service_code_dummied_df = pd.get_dummies(df['Service_Code'], drop_first=True)
    # df = pd.concat([df, service_code_dummied_df], axis=1)
    # df.drop('Service_Code', axis=1, inplace=True)
    #
    # proc_desc_dummied_df = pd.get_dummies(df['Procedure_Description'], drop_first=True)
    # df = pd.concat([df, proc_desc_dummied_df], axis=1)
    # df.drop('Procedure_Description', axis=1, inplace=True)
    #
    # fin_class_dummy_df = pd.get_dummies(df['Financial_Class'], drop_first=True)
    # df = pd.concat([df, fin_class_dummy_df], axis=1)
    # df.drop('Financial_Class', axis=1, inplace=True)
    #
    # nci_trans_det_dummy_df = pd.get_dummies(df['NCI_Transaction_Detail'], drop_first=True)
    # df = pd.concat([df, nci_trans_det_dummy_df], axis=1)
    # df.drop('NCI_Transaction_Detail', axis=1, inplace=True)
    #
    # admit_type_dummy_df = pd.get_dummies(df['Admit_Type'], drop_first=True)
    # df = pd.concat([df, admit_type_dummy_df], axis=1)
    # df.drop('Admit_Type', axis=1, inplace=True)
    #
    # cur_pat_desc_dummy_df = pd.get_dummies(df['Current_Patient_Class_Description'], drop_first=True)
    # df = pd.concat([df, cur_pat_desc_dummy_df], axis=1)
    # df.drop('Current_Patient_Class_Description', axis=1, inplace=True)
    #
    # dis_dept_dummy_df = pd.get_dummies(df['Discharge_Department_ID'], drop_first=True)
    # df = pd.concat([df, dis_dept_dummy_df], axis=1)
    # df.drop('Discharge_Department_ID', axis=1, inplace=True)
    #
    # prim_serv_name_dummy_df = pd.get_dummies(df['Primary_Service_Name'], drop_first=True)
    # df = pd.concat([df, prim_serv_name_dummy_df], axis=1)
    # df.drop('Primary_Service_Name', axis=1, inplace=True)
    #
    # region_dummy_df = pd.get_dummies(df['Region'], drop_first=True)
    # df = pd.concat([df, region_dummy_df], axis=1)
    # df.drop('Region', axis=1, inplace=True)
    #
    # # drop all dates here
    # df.drop(['Service_Date', 'Post_Date', 'Account_Discharge_Date', 'Account_Billed_Date',
    # 'Admit_Date', 'First_Claim_Date', 'Last_Claim_Date', 'Last_Payment_Date'], axis=1, inplace=True)
    #
    # # drop specific features here
    # df.drop(['Unnamed: 0', 'account_id', 'Transaction_Number', 'TransactionType_Tdata',
    #  'Procedure_Code', 'Insurance_Code', 'Insurance_Code_Description',
    #  'NCI_Transaction_Category', 'NCI_Transaction_Type', 'Account_Balance',
    #  'Admit_Diagnosis', 'Billing_Status', 'Medicaid_Pending_Flag', 'Medical_Record_Number',
    #  'Patient_Insurance_Balance', 'Patient_Self_Pay_Balance', 'Primary_Diagnosis',
    #  'ICD-10_Diagnosis_Principal', 'ICD-10_Diagnosis'], axis=1, inplace=True)
    #
    # # Create 6 different categories - one for each AWO bucket
    # df['AWO_Bucket'] = df['AWO_Bucket'].map({'1. <0': 0, '2. 0-1,000': 1,
    # '3. 1,000-2,500':2,  '4. 2,500-5,000':3, '5. 5,000-10,000':4, '6. 10,000+':5})
    #
    # df.to_csv('../data/trial_df_1.csv')

# Trial df 2
    df['AWO%'] = df['Transaction_Amount'] / df['NPSR']

    # Create Dummy Variables and Drop the following
    service_code_dummied_df = pd.get_dummies(df['Service_Code'], prefix='ServiceCode', drop_first=True)
    df = pd.concat([df, service_code_dummied_df], axis=1)
    df.drop('Service_Code', axis=1, inplace=True)

    nci_trans_det_dummy_df = pd.get_dummies(df['NCI_Transaction_Detail'], prefix='TransDetail', drop_first=True)
    df = pd.concat([df, nci_trans_det_dummy_df], axis=1)
    df.drop('NCI_Transaction_Detail', axis=1, inplace=True)

    cur_pat_desc_dummy_df = pd.get_dummies(df['Current_Patient_Class_Description'], prefix='ClassDescrp', drop_first=True)
    df = pd.concat([df, cur_pat_desc_dummy_df], axis=1)
    df.drop('Current_Patient_Class_Description', axis=1, inplace=True)

    dis_dept_dummy_df = pd.get_dummies(df['Discharge_Department_ID'], prefix='DisDept', drop_first=True)
    df = pd.concat([df, dis_dept_dummy_df], axis=1)
    df.drop('Discharge_Department_ID', axis=1, inplace=True)

    prim_serv_name_dummy_df = pd.get_dummies(df['Primary_Service_Name'], prefix='ServiceName', drop_first=True)
    df = pd.concat([df, prim_serv_name_dummy_df], axis=1)
    df.drop('Primary_Service_Name', axis=1, inplace=True)


    # drop all dates here
    df.drop(['Service_Date', 'Post_Date', 'Account_Discharge_Date', 'Account_Billed_Date',
    'Admit_Date', 'First_Claim_Date', 'Last_Claim_Date', 'Last_Payment_Date'], axis=1, inplace=True)

    # drop specific features here
    df.drop(['account_id', 'Transaction_Number', 'TransactionType_Tdata',
     'Procedure_Code', 'Insurance_Code', 'Insurance_Code_Description',
     'NCI_Transaction_Category', 'NCI_Transaction_Type', 'Account_Balance',
     'Admit_Diagnosis', 'Billing_Status', 'Medicaid_Pending_Flag', 'Medical_Record_Number',
     'Patient_Insurance_Balance', 'Patient_Self_Pay_Balance', 'Primary_Diagnosis',
     'ICD-10_Diagnosis_Principal', 'ICD-10_Diagnosis', 'LocationID', 'Financial_Class',
     'Admit_Type', 'Region', 'Procedure_Description', 'Transaction_Amount',
     'Last_Payment_Amount', 'Length_Of_Stay', 'NPSR'], axis=1, inplace=True)

    # Create 6 different categories - one for each AWO bucket
    df['AWO_Bucket'] = df['AWO_Bucket'].map({'1. <0': 0, '2. 0-1,000': 1,
    '3. 1,000-2,500':2,  '4. 2,500-5,000':3, '5. 5,000-10,000':4, '6. 10,000+':5})

    df.dropna(inplace=True)


    df.to_csv('../../../data/trial_df_2.csv')
