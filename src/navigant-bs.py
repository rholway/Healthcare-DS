import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_uniques_and_nan_count(df, column):
    print(column)
    print(f'Uniques: {df[column].unique()}')
    print(f'Number of Unique Values: {len(df[column].unique())}')
    print(f'Number of NaNs: {df[column].isna().sum()}')

def count_num_of_each_unique(df,column):
    print(df[column].value_counts())





if __name__ == '__main__':
    df = pd.read_excel('../data/scott-navigant.xlsx')
    df['Service_to_Post_Days'] = df['Post_Date'] - df['Service_Date']
    df = df[df['Financial_Division'] != 140034]

    df.drop(['Service_Area', 'Location_Name', 'Bill_Area', 'Department_Name',
    'Department_Number', 'Transaction_Type', 'Service_Date', 'Post_Date',
    'Procedure_Code', 'Financial_Class', 'Payor', 'CPT_Code_CPT_HCPCS',
    'CPT_Modifier_1_CPT_HCPCS', 'CPT_Modifier_2_CPT_HCPCS', 'CPT_Modifier_3_CPT_HCPCS',
    'CPT_Modifier_4_CPT_HCPCS', 'CPT_Modifier_5_CPT_HCPCS', 'Service_Provider',
    'Billing_Provider', 'Procedure_Description', 'Unnamed: 30', 'Unnamed: 31',
    0.3678623843877717], axis=1, inplace=True)

    df.rename(columns={'Gross Volume':'Gross_Volume', 'Net Volume': 'Net_Volume',
    'Final Net': 'Final_Net', 'Code Description': 'Code_Description'}, inplace=True)
    df['Void_Reversal_Status'] = df['Void_Reversal_Status'].map({'Y': 1, 'N': 0})
    df.drop(df.tail(1).index,inplace=True)
    df['Service_to_Post_Days'] = df['Service_to_Post_Days'].dt.days

    # print(get_uniques_and_nan_count(df, 'NCI_Transaction_Detail'))
    # print(count_num_of_each_unique(df, 'NCI_Transaction_Detail'))

    # indices = np.where(df['Financial_Division'].isna())
    # print(indices)

    print(df.shape)
