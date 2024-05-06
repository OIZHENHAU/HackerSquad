import numpy as np
import pandas as pd
import io
import requests
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbn
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, shapiro
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from LassoRegression import LassoRegressionCVModel

oilGas_csv = 'C:/Users/User/OneDrive/Desktop/WIA1006_Assignment/HackerSquad/oil_and_gas_data.csv';

oilGas_df = pd.read_csv(oilGas_csv)


# ################################# Impute DatasSet ################################################
def impute_dataset(df: pd.DataFrame):
    cols = df.columns

    for column in df.select_dtypes(include=['number']).columns:
        mean_value = df[column].mean()

        df[column].fillna(mean_value, inplace=True)


impute_dataset(oilGas_df)


# ################################ REMOVE OUTLIERS #################################################

def check_outlier(df: pd.DataFrame):
    curr_df = df
    cols = curr_df.columns

    curr_df.boxplot(patch_artist=True)
    plt.show()


check_outlier(oilGas_df)


def remove_outliers(df: pd.DataFrame):
    curr_df = df
    cols = curr_df.columns

    for i in range(len(cols)):
        col = cols[i]

        if col == 'Close':
            continue

        if curr_df[col].dtypes == 'object':
            continue

        Q1 = curr_df[col].quantile(0.25)
        Q3 = curr_df[col].quantile(0.75)

        IQR = Q3 - Q1

        Lower_Range = Q1 - (1.5 * IQR)
        Upper_Range = Q3 + (1.5 * IQR)

        # count = ((curr_df[col] < Lower_Range) | (curr_df[col] > Upper_Range)).sum()
        curr_df = curr_df[~((curr_df[col] < Lower_Range) | (curr_df[col] > Upper_Range))]

    return curr_df


oilGas_df = remove_outliers(oilGas_df)
# print(oilGas_df)

# ##################################### LABEL ENCODING ########################################################
label_encoder = LabelEncoder()

data_column_category = oilGas_df.select_dtypes(exclude=[np.number]).columns

for category_column in data_column_category:
    oilGas_df[category_column] = label_encoder.fit_transform(oilGas_df[category_column])


# #################################### PROCESSING DATA #########################################################
def processing_and_scale_data(df: pd.DataFrame):
    X = df[['Symbol', 'Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    minmax_scaler = preprocessing.MinMaxScaler()
    X_train = minmax_scaler.fit_transform(X_train)
    X_test = minmax_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = processing_and_scale_data(oilGas_df)
lassoCVModel = LassoRegressionCVModel(0.01, 1, 0.01, 10, 3)

lassoCVModel.fit(X_train, y_train)

prediction = lassoCVModel.predict(X_test)

plt.scatter(y_test, prediction)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Values (r = {0:0.2f})'.format(pearsonr(y_test, prediction)[0], 2))
plt.show()

# ##################################### PERFORM METRICS ######################################################

metrics_df = pd.DataFrame({'Metric':
                               ['MAE',
                                'MSE',
                                'RMSE',
                                'R-Squared'], 'Value':
                               [metrics.mean_absolute_error(y_test, prediction),
                                metrics.mean_squared_error(y_test, prediction),
                                np.sqrt(metrics.mean_squared_error(y_test, prediction)),
                                metrics.explained_variance_score(y_test, prediction)]}).round(3)
print(metrics_df)
print()


