import os
import tarfile
import urllib.request as req
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
import numpy as np
%matplotlib inline

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
HOUSING_PATH = os.path.join('Data')
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    req.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(housing_path)
    housing_tgz.close()


def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

fetch_housing_data()

housing = load_housing_data()

housing.info()

housing.hist(bins=50, figsize=(20,15))
plt.show()

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)

housing['income_cat'] = pd.cut(housing['median_income'], bins=[0.,1.5,3.0,4.5,6., np.inf], labels=[1,2,3,4,5])
housing['income_cat'].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]

start_train_set['income_cat'].value_counts()/len(start_train_set)

for set_ in (start_train_set, start_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

housing = start_train_set.copy()

housing.plot(kind='scatter', x='longitude', y='latitude')
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing['population']/100, label='population', figsize=(10,7), c='median_house_value', cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12,8)

housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)

housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_household'] = housing['total_bedrooms']/housing['households']
housing['population_per_household'] = housing['population']/housing['households']

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

housing = start_train_set.drop('median_house_value', axis=1)
housing_labels= start_train_set["median_house_value"].copy()


# 누락 데이터 채우는 방법 3가지

housing.dropna(subset=['total_bedrooms']) # option 1
housing.drop('total_bedrooms', axis=1) # option 2
# option 3
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median, inplace=True)