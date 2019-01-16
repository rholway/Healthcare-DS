import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as scs


def make_QQ_plot(y_actual, y_predicted, title='QQ Plot'):
    residuals = y_actual - y_predicted
    sm.graphics.qqplot(residuals, line='45', fit=True)
    plt.title(title)
    # plt.show()
    # plt.savefig('../../images/ryan_imgs/QQ-linreg')

def plot_residuals(y_actual, y_predicted, x_label, y_label='Residuals', title='Plot of Residuals'):
    '''
    Scatter plot of residuals
    y_actual: numpy array - true targets
    y_predicted: numpy array - predicted targets from model
    x_label: str - name of x label
    y_label: str - name of y label
    title: str - title
    y_actual and y_predicted must be same length
    '''
    residuals = y_actual - y_predicted
    fig, ax = plt.subplots(figsize=(12,4))
    ax.scatter(y_predicted, residuals, alpha=.1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.axhline(0,c='r',linestyle='--')
    plt.savefig('../../images/ryan_imgs/resids-linreg')
    # plt.close()
    # plt.show()

if __name__ == '__main__':
    # Load the dataset
    # df1 = pd.read_excel('../../../data/DemoData.xlsx')
    # df = pd.read_csv('../../../data/sample_df')
    df = pd.read_csv('../../../data/trial_df_2.csv')


    df.drop(['Unnamed: 0', 'AWO_Bucket', 'ServiceCode_OUTPATIENT', 'DisDept_OBSERVATION',
    'DisDept_REHABILITATION', 'DisDept_RENAL', 'DisDept_URGENT', 'ServiceName_ OBSERVATION PATIENT',
    'ServiceName_ OBSERVATION REMICAID', 'ServiceName_ URGENT CARE CENTER', 'ServiceName_OBSERVATION PATIENT',
    'ServiceName_OBSERVATION SDC PLANNED', 'ServiceName_OUTPATIENT REHAB', 'ServiceName_PACT SERVICES',
    'ServiceName_REC or OUT RENAL DIALYSIS', 'ServiceName_RECURRING REHAB', 'ServiceName_URGENT CARE',
    'ServiceCode_EMERGENCY',
     'ServiceCode_INPATIENT',
     'TransDetail_Non Covered',
     'ClassDescrp_INPATIENT',
     'ClassDescrp_OFFICE VISIT',
     'DisDept_EMERGENCY',
     'DisDept_ENDOCRONIC',
     'DisDept_HEMOGLOBIN',
     'DisDept_NUEROLOGY',
     'DisDept_PSYCH',
     'DisDept_PULMONARY',
     'DisDept_SURGERY',
     'ServiceName_EMERGENCY ROOM PATIENT',
     'ServiceName_None',
     'ServiceName_OUTPATIENT',
     'ServiceName_OUTPATIENT PSYCH',
     'ServiceName_RECURRING PSYCH'], axis=1, inplace=True)


    df.dropna(inplace=True)
    y = (df.pop('AWO%').values) * 100
    X = df.values


    # Use only one feature
    # diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train,y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)
    '''
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.6f"
          % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))

    # For each X, calculate VIF and save in dataframe
    vif = pd.DataFrame()
    vif["VIF_Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif["features"] = df.columns
    print(vif.round(1))
    '''
    ols_model = sm.OLS(endog=y, exog=X).fit()
    # make_QQ_plot(y_test, y_pred, 'QQ plot')
    plot_residuals(y_test, y_pred, 'Predicted AWO %')
