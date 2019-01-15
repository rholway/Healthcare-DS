import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def make_bar_graph(df, col, title, filename):
    '''
    Input: DataFrame, Column from that DataFrame
    Output: Bar Graph
    The bar graph will be a breakdown of each of the unique values within the
    column.  Each unique value will have an associated percentage, which is the
    AWO Transaction_Amount as a percentage of NPSR.
    '''
    new_df = df[[col, 'Transaction_Amount', 'NPSR']]
    gb_df = new_df.groupby(col).sum().reset_index(drop=False)
    gb_df['Percentage'] = (gb_df['Transaction_Amount'] / gb_df['NPSR'])*100
    gb_df.loc['TOTAL'] = ['TOTAL', gb_df['Transaction_Amount'].sum(),gb_df['NPSR'].sum(),(gb_df['Transaction_Amount'].sum()/gb_df['NPSR'].sum()*100)]
    # plot
    x = np.arange(gb_df.shape[0])
    percentages = list(gb_df['Percentage'].values)
    groups = list(gb_df[col].values)
    fig, ax = plt.subplots(figsize=(8,8))
    fig.subplots_adjust(bottom=0.4)
    plt.ylabel('AWO as % of NPSR', fontsize=12)
    plt.bar(x, percentages, color='b')
    plt.xticks(x, groups, rotation=90, fontsize=12)
    plt.title(title, fontsize=14)
    rects = ax.patches
    for rect in rects:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        space = 0
        va = 'bottom'
        if y_value < 0:
            space *= -1
            va = 'top'
        label = "{:.2f}".format(y_value)
        plt.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.
    plt.savefig('../images/hosp_6/' + filename)
    # plt.show()



if __name__ == '__main__':
    # df = pd.read_csv('../../data/sample_df')
    df = pd.read_excel('../../data/DemoData.xlsx')

    # df1 = df[['Insurance_Code_Description', 'Transaction_Amount', 'NPSR']]
    # df2 = df1.groupby('Insurance_Code_Description').sum()
    # df2['Percentage'] = (df2['Transaction_Amount']/df2['NPSR'])*100

# All hospitals
    # bar plot of grouped-by insurance code description for all locations
    # print(make_bar_graph(df, 'Insurance_Code_Description',
    # 'Grouped By Insurance Code Description', 'all-insurance'))

    # bar plot of grouped-by transaction detail for all locations
    # print(make_bar_graph(df, 'NCI_Transaction_Detail',
    # 'Grouped By Transaction Detail', 'all-trans-det'))

    # bar plot of grouped-by locationid for all locations
    # print(make_bar_graph(df, 'LocationID',
    # 'Grouped By Hospital', 'all-loc'))

    # bar plot of grouped-by patient class for all locations
    # print(make_bar_graph(df, 'Financial_Class',
    # 'Grouped By Patient Class', 'all-fin-class'))

    # bar plot of grouped-by department for all locations
    # print(make_bar_graph(df, 'Discharge_Department_ID',
    # 'Grouped By Department', 'all-dept'))

    # bar plot of grouped-by service type for all locations
    # print(make_bar_graph(df, 'Service_Code',
    # 'Grouped By Patient Class', 'all-patient-class'))

# Hospital 1
    # h1_df = df[df['LocationID'] == 'Hospital1']

    # bar plot of grouped-by insurance code description for H1
    # print(make_bar_graph(h1_df, 'Insurance_Code_Description',
    # 'Hospital 1 Grouped By Insurance Code Description', 'h1-insurance'))

    # bar plot of grouped-by transaction detail for H1
    # print(make_bar_graph(h1_df, 'NCI_Transaction_Detail',
    # 'Hospital 1 Grouped By Transaction Detail', 'h1-trans-det'))

    # bar plot of grouped-by department for H1
    # print(make_bar_graph(h1_df, 'Discharge_Department_ID',
    # 'Hospital 1 Grouped By Department', 'h1-dept'))

    # bar plot of grouped-by service type for H1
    # print(make_bar_graph(h1_df, 'Service_Code',
    # 'Hospital 1 Grouped By Patient Class', 'h1-patient-class'))

# Hospital 2
    # h2_df = df[df['LocationID'] == 'Hospital2']

    # bar plot of grouped-by insurance code description for H2
    # print(make_bar_graph(h2_df, 'Insurance_Code_Description',
    # 'Hospital 2 Grouped By Insurance Code Description', 'h2-insurance'))

    # bar plot of grouped-by transaction detail for H2
    # print(make_bar_graph(h2_df, 'NCI_Transaction_Detail',
    # 'Hospital 2 Grouped By Transaction Detail', 'h2-trans-det'))

    # bar plot of grouped-by department for H2
    # print(make_bar_graph(h2_df, 'Discharge_Department_ID',
    # 'Hospital 2 Grouped By Department', 'h2-dept'))

    # bar plot of grouped-by service type for H2
    # print(make_bar_graph(h2_df, 'Service_Code',
    # 'Hospital 2 Grouped By Patient Class', 'h2-patient-class'))

# Hospital 3
    # h3_df = df[df['LocationID'] == 'Hospital3']

    # bar plot of grouped-by insurance code description for H3
    # print(make_bar_graph(h3_df, 'Insurance_Code_Description',
    # 'Hospital 3 Grouped By Insurance Code Description', 'h3-insurance'))

    # bar plot of grouped-by transaction detail for H3
    # print(make_bar_graph(h3_df, 'NCI_Transaction_Detail',
    # 'Hospital 3 Grouped By Transaction Detail', 'h3-trans-det'))

    # bar plot of grouped-by department for H3
    # print(make_bar_graph(h3_df, 'Discharge_Department_ID',
    # 'Hospital 3 Grouped By Department', 'h3-dept'))

    # bar plot of grouped-by service type for H3
    # print(make_bar_graph(h3_df, 'Service_Code',
    # 'Hospital 3 Grouped By Patient Class', 'h3-patient-class'))

# Hospital 4
    # h4_df = df[df['LocationID'] == 'Hospital4']

    # bar plot of grouped-by insurance code description for H4
    # print(make_bar_graph(h4_df, 'Insurance_Code_Description',
    # 'Hospital 4 Grouped By Insurance Code Description', 'h4-insurance'))

    # bar plot of grouped-by transaction detail for H4
    # print(make_bar_graph(h4_df, 'NCI_Transaction_Detail',
    # 'Hospital 4 Grouped By Transaction Detail', 'h4-trans-det'))

    # bar plot of grouped-by department for H4
    # print(make_bar_graph(h4_df, 'Discharge_Department_ID',
    # 'Hospital 4 Grouped By Department', 'h4-dept'))

    # bar plot of grouped-by service type for H4
    # print(make_bar_graph(h4_df, 'Service_Code',
    # 'Hospital 4 Grouped By Patient Class', 'h4-patient-class'))

# Hospital 5
    # h5_df = df[df['LocationID'] == 'Hospital5']

    # bar plot of grouped-by insurance code description for H5
    # print(make_bar_graph(h5_df, 'Insurance_Code_Description',
    # 'Hospital 5 Grouped By Insurance Code Description', 'h5-insurance'))

    # bar plot of grouped-by transaction detail for H5
    # print(make_bar_graph(h5_df, 'NCI_Transaction_Detail',
    # 'Hospital 5 Grouped By Transaction Detail', 'h5-trans-det'))

    # bar plot of grouped-by department for H5
    # print(make_bar_graph(h5_df, 'Discharge_Department_ID',
    # 'Hospital 5 Grouped By Department', 'h5-dept'))

    # bar plot of grouped-by service type for H5
    # print(make_bar_graph(h5_df, 'Service_Code',
    # 'Hospital 5 Grouped By Patient Class', 'h5-patient-class'))

# Hospital 6
    # h6_df = df[df['LocationID'] == 'Hospital6']

    # bar plot of grouped-by insurance code description for H6
    # print(make_bar_graph(h6_df, 'Insurance_Code_Description',
    # 'Hospital 6 Grouped By Insurance Code Description', 'h6-insurance'))

    # bar plot of grouped-by transaction detail for H6
    # print(make_bar_graph(h6_df, 'NCI_Transaction_Detail',
    # 'Hospital 6 Grouped By Transaction Detail', 'h6-trans-det'))

    # bar plot of grouped-by department for H6
    # print(make_bar_graph(h6_df, 'Discharge_Department_ID',
    # 'Hospital 6 Grouped By Department', 'h6-dept'))

    # bar plot of grouped-by service type for H6
    # print(make_bar_graph(h6_df, 'Service_Code',
    # 'Hospital 6 Grouped By Patient Class', 'h6-patient-class'))
