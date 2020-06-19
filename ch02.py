from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
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


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    req.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


fetch_housing_data()

housing = load_housing_data()

housing.info()

housing.hist(bins=50, figsize=(20, 15))
plt.show()

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing['income_cat'] = pd.cut(housing['median_income'], bins=[
                               0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
housing['income_cat'].hist()

# stratifiedkhold + ShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# 계층 샘플링
for train_index, test_index in split.split(housing, housing['income_cat']):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]

# 비율 살펴보기
start_train_set['income_cat'].value_counts()/len(start_train_set)

# income_cat 삭제 후 데이터 리셋
for set_ in (start_train_set, start_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

# 훈련세트 복사
housing = start_train_set.copy()

# 지리적 데이터 시각화
housing.plot(kind='scatter', x='longitude', y='latitude')

# alpha = 투명도
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)

# 컬러맵 정의
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='population', figsize=(10, 7), c='median_house_value', cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()

# 표준 상관계수
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

# 특성 그래픽화
attributes = ['median_house_value', 'median_income',
              'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 8))

# 중간 주택가격과 중간 소득 상관관계 산점도 x = 소득  y = 주택가격
housing.plot(kind='scatter', x='median_income',
             y='median_house_value', alpha=0.1)

# 특성조합
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_household'] = housing['total_bedrooms'] / \
    housing['total_rooms']
housing['population_per_household'] = housing['population'] / \
    housing['households']

# 표준 상관ㄱㅖ수
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

# 데이터셋 다시 준비
housing = start_train_set.drop('median_house_value', axis=1)
housing_labels = start_train_set["median_house_value"].copy()


# 누락 데이터 채우는 방법 3가지

housing.dropna(subset=['total_bedrooms'])  # option 1
housing.drop('total_bedrooms', axis=1)  # option 2
# option 3
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median, inplace=True)


# 사이킷런에서 누락된 값을 손쉽게 다루게함
imputer = SimpleImputer(strategy="median")

# Object 속성인 ocean_proximity 제외한 데이터 복사본 생성
housing_sum = housing.drop('ocean_proximity', axis=1)

# 데이터 훈련
imputer.fit(housing_sum)
# 각 특성의 중간값을 계산해서 그 결과를 statistics_ 객체에 저장
# 시스템이 서비스 될때 어떤 데이터가 누락될지 확신 할 수 없어서 모든 수치형 특성에 imputer를 적용하는것이 바람직함.
imputer.statistics_
housing_sum.median().values

# 누락된 곳을 중간 값으로 채운 것을 다시 데이터프레임으로 만듬 X는 평범한 넘파이 배열이다.
X = imputer.transform(housing_sum)
housing_tr = pd.DataFrame(
    X, columns=housing_sum.columns, index=housing_sum.index)

housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)

# 텍스트를 숫자로 변형
ordinal = OrdinalEncoder()
housing_cat_encoded = ordinal.fit_transform(housing_cat)
housing_cat_encoded[:10]

# 인스턴스 변수로 카테고리 목록 열 수 있음.
ordinal.categories_

# One-hot Encoding
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
housing_cat_1hot.toarray()

cat_encoder.categories_

lf = pd.DataFrame(housing_cat_1hot.toarray(), columns=cat_encoder.categories_)
lf


# 나만의 변환기
from sklearn.base import BaseEstimator, TransformerMixin
# index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAddr(BaseEstimator, TransformerMixin):
    # *args나 **kargs가 아닙니다.
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self         # 더이상 할일이 없습니다.

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAddr(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# 변환 파이프라인

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('attribs_adder', CombinedAttributesAddr()),('std_scaler', StandardScaler())])

housing_num_tr = num_pipeline.fit_transform(housing_sum)

housing_num_tr

# 하나의 변화기롤 각 열마다 벅절히 변환 적용 처리

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_sum)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([('num', num_pipeline, num_attribs), ('cat', OneHotEncoder(), cat_attribs)])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
num_attribs

# 모델선택 과 훈련

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)
lin_reg.predict(some_data_prepared)
list(some_labels)

# 과소적합
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# 과대 적합
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# 교차검증
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)

tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print('점수: ', scores)
    print('average : ', scores.mean())
    print('STD : ', scores.std())

display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)

forest_rmse =  np.sqrt(-forest_scores)
display_scores(forest_rmse)



# 모델 저장
import joblib
joblib.dump(CombinedAttributesAddr, 'my_model.pkl')

# 모델 불러오기
my_model_loaded =joblib.load('my_model.pkl')