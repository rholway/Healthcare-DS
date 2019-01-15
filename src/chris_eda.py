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

def plot_agg_pie(df, group_col, target_col, aggregate_func, y_label, title, save_fig=None):
    fig, ax = plt.subplots()
    grouped = df.groupby(group_col).aggregate({target_col : aggregate_func})
    x_range = list(range(0, grouped.shape[0]))
    x_labels = grouped.index.tolist()
    y_vals = grouped.values.reshape(1,-1)[0]
    ax.pie(y_vals, radius=1, labels=x_labels, autopct='%1.1f%%')
    center_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(center_circle)
    ax.axis('equal')
    ax.set_title(title)
    plt.tight_layout()
    if save_fig:
        file_name = title.replace(' ', '_') + '.png'
        plt.savefig(save_fig_path.format(file_name))
    else:
        plt.show()
        plt.close()

def pie_2(df, group_col, target_col, aggregate_func, y_label, title, save_fig=None):
    grouped = df.groupby(group_col).aggregate({target_col : aggregate_func})
    x_range = list(range(0, grouped.shape[0]))
    x_labels = grouped.index.tolist()
    y_vals = grouped.values.reshape(1,-1)[0]
    fig, axs = plt.subplots(1,2, figsize=(9,5))
    axs = axs.ravel()
    for i in range(2):
        axs[i].pie(y_vals, radius=1, labels=x_labels, autopct='%1.1f%%')
        center_circle = plt.Circle((0,0),0.70,fc='white')
        # fig = plt.gcf()
        fig.gca().add_artist(center_circle)
        # axs[i].add_artist(plt.Circle((0,0),0.70,fc='white'))
        axs[i].axis('equal')
        axs[i].set_title(title)
        plt.tight_layout()
    if save_fig:
        file_name = title.replace(' ', '_') + '.png'
        plt.savefig(save_fig_path.format(file_name))
    else:
        plt.show()
        plt.close()


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

    df_outpatient = df[df['Service_Code'] == 'OUTPATIENT']
    df_emergency = df[df['Service_Code'] == 'EMERGENCY']
    df_observation = df[df['Service_Code'] == 'OBSERVATION']
    df_dialysis = df[df['Service_Code'] == 'DIALYSIS']
    df_psych = df[df['Service_Code'] == 'PSYCH']
    df_rehab = df[df['Service_Code'] == 'REHAB']

    # next_group = df.groupby(['Service_Code', 'NCI_Transaction_Detail'])['target_percentage'].sum()

    # df_full = pd.read_csv('../../navigant_data/final_df_cl_edit.csv')

    # plot_agg_pie(df, 'Service_Code', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Total percent AWO of NPSR by Service Code', save_fig=True)
    # plot_agg_pie(df, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)
    #
    # plot_agg_pie(loc_1, 'Service_Code', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Hospital 1: Total percent AWO of NPSR by Service Code', save_fig=True)
    # plot_agg_pie(loc_1, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Hospital 1: Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)
    #
    # plot_agg_pie(loc_2, 'Service_Code', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Hospital 2: Total percent AWO of NPSR by Service Code', save_fig=True)
    # plot_agg_pie(loc_2, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Hospital 2: Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)
    #
    # plot_agg_pie(loc_3, 'Service_Code', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Hospital 3: Total percent AWO of NPSR by Service Code', save_fig=True)
    # plot_agg_pie(loc_3, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Hospital 3: Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)
    #
    # plot_agg_pie(loc_4, 'Service_Code', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Hospital 4: Total percent AWO of NPSR by Service Code', save_fig=True)
    # plot_agg_pie(loc_4, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Hospital 4: Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)
    #
    # plot_agg_pie(loc_5, 'Service_Code', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Hospital 5: Total percent AWO of NPSR by Service Code', save_fig=True)
    # plot_agg_pie(loc_5, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Hospital 5: Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)
    #
    # plot_agg_pie(loc_6, 'Service_Code', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Hospital 6: Total percent AWO of NPSR by Service Code', save_fig=True)
    # plot_agg_pie(loc_6, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Hospital 6: Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)

    plot_agg_pie(df_outpatient, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Outpatient: Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)
    plot_agg_pie(df_emergency, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Emergency: Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)
    plot_agg_pie(df_observation, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Observation: Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)
    plot_agg_pie(df_dialysis, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Dialysis: Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)
    plot_agg_pie(df_psych, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Psych: Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)
    plot_agg_pie(df_rehab, 'NCI_Transaction_Detail', 'target_percentage', 'sum', 'Percent AWO of NPSR', 'Rehab: Total percent AWO of NPSR by NCI_Transaction_Detail', save_fig=True)
