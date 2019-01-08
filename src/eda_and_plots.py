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
    plt.bar(x, percentages)
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
    plt.savefig('../images/all_hospitals/' + filename)



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
    # 'Grouped By Service Type', 'all-serv-type'))

# Hospital 1
    
    # bar plot of grouped-by insurance code description for H1
    print(make_bar_graph(df, 'Insurance_Code_Description',
    'Grouped By Insurance Code Description', 'all-insurance'))

    # bar plot of grouped-by transaction detail for H1
    print(make_bar_graph(df, 'NCI_Transaction_Detail',
    'Grouped By Transaction Detail', 'all-trans-det'))

    # bar plot of grouped-by locationid for H1
    print(make_bar_graph(df, 'LocationID',
    'Grouped By Hospital', 'all-loc'))

    # bar plot of grouped-by patient class for H1
    print(make_bar_graph(df, 'Financial_Class',
    'Grouped By Patient Class', 'all-fin-class'))

    # bar plot of grouped-by department for H1
    print(make_bar_graph(df, 'Discharge_Department_ID',
    'Grouped By Department', 'all-dept'))

    # bar plot of grouped-by service type for H1
    print(make_bar_graph(df, 'Service_Code',
    'Grouped By Service Type', 'all-serv-type'))
