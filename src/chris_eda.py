import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

def plot_agg_grouped(df, group_col, target_col, aggregate_func, y_label, title, save_fig=None):
    # pdb.set_trace()
    grouped = df.groupby(group_col).aggregate({target_col : aggregate_func})
    x_range = list(range(0, grouped.shape[0]))
    x_labels = grouped.index.tolist()
    y_vals = grouped.values.reshape(1,-1)[0]
    plt.bar(x_range, y_vals, color='b')
    plt.xticks(x_range, x_labels, rotation=45, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    if save_fig:
        file_name = title.replace(' ', '_') + '.png'
        plt.savefig(save_fig_path.format(file_name))
    else:
        plt.show()
        plt.close()

def plot_double_grouped():
    pass

if __name__=='__main__':

    plt.rcParams["figure.figsize"] = (8,8)
    save_fig_path = '../images/chris_imgs/{}'

    df = pd.read_excel('../../navigant_data/DemoData.xlsx')
    df_short = df.iloc[:100]
    df['target_percentage'] = df['account_id.1'] / df['NPSR']
    loc_1 = df[df['LocationID'] == 'Hospital1']
    loc_2 = df[df['LocationID'] == 'Hospital2']
    loc_3 = df[df['LocationID'] == 'Hospital3']
    loc_4 = df[df['LocationID'] == 'Hospital4']
    loc_5 = df[df['LocationID'] == 'Hospital5']
    loc_6 = df[df['LocationID'] == 'Hospital6']

    # next_group = df.groupby(['Service_Code', 'NCI_Transaction_Detail'])['target_percentage'].sum()

    # df_full = pd.read_csv('../../navigant_data/final_df_cl_edit.csv')

    # plot_agg_grouped(df, 'Service_Code', 'account_id.1', 'sum')
    # plot_agg_grouped(df, 'NCI_Transaction_Detail', 'account_id.1', 'sum')
    # plot_agg_grouped(df, 'Service_Code', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Total AWO/NPSR percent by Service Code')
    plot_agg_grouped(df, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)
