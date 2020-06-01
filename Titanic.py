#%%
# 필요한 모듈 로드
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
from sklearn.metrics import accuracy_score

# 전체보기
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_row', None)

# Loading the datalo
train = pd.read_csv("C:\\Users\\wdp\\titanic\\train.csv")
test = pd.read_csv("C:\\Users\\wdp\\titanic\\test.csv")

# 나이 예측 모델을 위한 준비
# train에 있는 'Survived' 컬럼을 삭제하고, train과 test를 concat
temp_all = pd.concat([train.drop('Survived', axis = 1), test], ignore_index = True, axis=0)

# 'Age'에 누락데이터 행을 삭제
temp_all.dropna(subset=['Age'], axis=0, inplace=True)

#%%
# 나이예측 모델 함수화
def age_prediction(temp_for_age):
    # 나이 예측 모델을 학습시키기 위한 전처리
    # 결측치 확인
    # temp_for_age.isnull().sum( axis=0 )
    """
    PassengerId       0
    Pclass            0
    Name              0
    Sex               0
    Age             263
    SibSp             0
    Parch             0
    Ticket            0
    Fare              1
    Cabin          1014
    Embarked          2
    """
    # 이름을 매핑하고, 나이별로 구분
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""
    
    temp_for_age['Title'] = temp_for_age['Name'].apply(get_title)
    # pd.Series.value_counts(temp_for_age['Title'])
    """
    Mr          757
    Miss        260
    Mrs         197
    Master       61
    Rev           8
    Dr            8
    Col           4
    Mlle          2
    Ms            2
    Major         2
    Jonkheer      1
    Mme           1
    Countess      1
    Capt          1
    Dona          1
    Lady          1
    Don           1
    Sir           1
    """
    # Title별 평균나이를 순서대로 정렬
    # temp_for_age[['Title', 'Age']].groupby(['Title'], as_index=False).mean().sort_values(by='Age', ascending=True)
    """
    9     Master   5.482642
    10      Miss  21.774238
    11      Mlle  24.000000
    12       Mme  24.000000
    15        Ms  28.000000
    13        Mr  32.252151
    2   Countess  33.000000
    14       Mrs  36.994118
    6   Jonkheer  38.000000
    4       Dona  39.000000
    3        Don  40.000000
    16       Rev  41.250000
    5         Dr  43.571429
    7       Lady  48.000000
    8      Major  48.500000
    17       Sir  49.000000
    1        Col  54.000000
    0       Capt  70.000000
    """
    # 나이가 많은 순대로 높은 수로 mapping
    temp_for_age['Title'] = temp_for_age['Title'].replace(['Master'], 0)
    temp_for_age['Title'] = temp_for_age['Title'].replace(['Miss','Mme','Mlle','Ms'], 1)
    temp_for_age['Title'] = temp_for_age['Title'].replace(['Mr','Countess','Mrs','Jonkheer','Dona','Don','Rev'], 2)
    temp_for_age['Title'] = temp_for_age['Title'].replace(['Dr','Lady','Major','Sir','Col'], 3)
    temp_for_age['Title'] = temp_for_age['Title'].replace(['Capt'], 4)
        
    # Cabin 유무에 따른 나이 확인
    temp_for_age['Has_Cabin'] = temp_for_age["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    temp_for_age[['Has_Cabin', 'Age']].groupby(['Has_Cabin'], as_index=False).mean().sort_values(by='Has_Cabin', ascending=True)
    """
       Has_Cabin        Age
    0          0  27.406654
    1          1  36.922500
    """
    # FamilySize에 따른 나이 확인
    temp_for_age['FamilySize'] = temp_for_age['SibSp'] + temp_for_age['Parch'] + 1
    
    temp_for_age[['Pclass','Age']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Pclass', ascending=True)
    """
       Pclass        Age
    0       1  39.159930
    1       2  29.506705
    2       3  24.816367
    """
    mask_temp_for_age_1 = (temp_for_age.FamilySize >= 3) | (temp_for_age.Pclass =='2')
    temp_for_age['2_family'] = mask_temp_for_age_1.astype(int)
    
    mask_temp_for_age_2 = (temp_for_age.FamilySize >= 3) | (temp_for_age.Pclass =='3')
    temp_for_age['3_family'] = mask_temp_for_age_2.astype(int)
    
    temp_for_age['IsAlone'] = 0
    temp_for_age.loc[temp_for_age['FamilySize'] == 1, 'IsAlone'] = 1
    
    # pd.Series.value_counts(temp_for_age['FamilySize'])
    """
    1     790
    2     235
    3     159
    4      43
    6      25
    5      22
    7      16
    11     11
    8       8
    """
    # temp_for_age[['FamilySize', 'Age']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Age', ascending=True)
    """
       FamilySize        Age
    8          11  14.500000
    6           7  17.375000
    7           8  18.000000
    3           4  19.423333
    5           6  20.120000
    4           5  23.764706
    2           3  26.534097
    0           1  31.511864
    1           2  32.726942
    """
    
    temp_for_age['FamilySize'] = temp_for_age['FamilySize'].replace({11:1,7:1,8:1,4:1,6:1,5:2,3:2,1:3,2:3})
    temp_for_age[['FamilySize', 'Age']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Age', ascending=True)
    
    temp_for_age['Embarked'] = temp_for_age['Embarked'].fillna('S')
    temp_for_age['Embarked'] = temp_for_age['Embarked'].map( {'S': 1, 'C': 2, 'Q': 0} )
    
    temp_for_age[['Embarked', 'Age']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Age', ascending=True)
    """
       Embarked        Age
    2         2  28.630000
    0         0  29.298151
    1         1  32.332170
    """
    temp_for_age['Sex'] = temp_for_age['Sex'].map( {'female': 0, 'male': 1} )
    temp_for_age[['Sex', 'Age']].groupby(['Sex'], as_index=False).mean().sort_values(by='Age', ascending=True)
    """
       Sex        Age
    0    0  28.687088
    1    1  30.585228
    """

    temp_for_age['Fare'] = temp_for_age['Fare'].fillna(temp_for_age['Fare'].median())
    temp_for_age.loc[ temp_for_age['Fare'] <= 4, 'Fare'] 		        = 0
    temp_for_age.loc[(temp_for_age['Fare'] > 4) & (temp_for_age['Fare'] <= 8), 'Fare']   = 1
    temp_for_age.loc[(temp_for_age['Fare'] > 8) & (temp_for_age['Fare'] <= 12), 'Fare']   = 2
    temp_for_age.loc[(temp_for_age['Fare'] > 12) & (temp_for_age['Fare'] <= 16), 'Fare']   = 3
    temp_for_age.loc[(temp_for_age['Fare'] > 16) & (temp_for_age['Fare'] <= 20), 'Fare']   = 4
    temp_for_age.loc[(temp_for_age['Fare'] > 20) & (temp_for_age['Fare'] <= 24), 'Fare']   = 5
    temp_for_age.loc[(temp_for_age['Fare'] > 24) & (temp_for_age['Fare'] <= 28), 'Fare']   = 6
    temp_for_age.loc[(temp_for_age['Fare'] > 28) & (temp_for_age['Fare'] <= 32), 'Fare']   = 7
    temp_for_age.loc[(temp_for_age['Fare'] > 32) & (temp_for_age['Fare'] <= 36), 'Fare']   = 8
    temp_for_age.loc[(temp_for_age['Fare'] > 36) & (temp_for_age['Fare'] <= 40), 'Fare']   = 9
    temp_for_age.loc[(temp_for_age['Fare'] > 40) & (temp_for_age['Fare'] <= 44), 'Fare']   = 10
    temp_for_age.loc[(temp_for_age['Fare'] > 44) & (temp_for_age['Fare'] <= 48), 'Fare']   = 11
    temp_for_age.loc[ temp_for_age['Fare'] > 48, 'Fare'] 		 = 12
    temp_for_age['Fare'] = temp_for_age['Fare'].astype(int)
    
    # temp_for_age[['Fare', 'Age']].groupby(['Fare'], as_index=False).mean().sort_values(by='Age', ascending=True)
    """
        Fare        Age
    11    11  19.909091
    4      4  19.977667
    9      9  23.360000
    5      5  23.416739
    10    10  25.142857
    2      2  26.876645
    1      1  27.535088
    7      7  27.663333
    3      3  29.196181
    0      0  33.777778
    6      6  34.632294
    8      8  35.235294
    12    12  36.607926
    """
    temp_for_age['Fare'] = temp_for_age['Fare'].replace({11:0,4:0,9:1,5:1,10:1,2:1,1:1,7:2,3:2,0:3,6:3,8:3,12:3})
    temp_for_age[['Fare', 'Age']].groupby(['Fare'], as_index=False).mean().sort_values(by='Age', ascending=True)
    
    # 필요없는 컬럼 삭제
    # temp_for_age.columns
    # 'PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Title','Has_Cabin','FamilySize','2_family','3_family','IsAlone']

    drop_elements = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket','Cabin']
    temp_for_age = temp_for_age.drop(drop_elements, axis = 1)

    # 컬럼별 상관관계
    colormap = plt.cm.viridis
    plt.figure(figsize=(10,10))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(temp_for_age.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
    return temp_for_age   
#%%
temp_all = age_prediction(temp_all)
temp_all.describe()
#%%

X_age = temp_all[ ['Pclass','Fare', 'Title', 'Has_Cabin',
       'FamilySize','IsAlone'] ]  # 독립변수
y_age = temp_all['Age']  # 종속변수

# train data 와 test data로 구분
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_age, y_age, test_size=0.3, random_state=10) 

print('train data 개수: ', X_train.shape)
print('test data 개수: ', X_test.shape)
print('\n')

from sklearn import metrics
from xgboost import XGBRFRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

def train_and_test(model):
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    report = metrics.mean_squared_error(y_test, y_hat)            
    print(report)

    return y_hat

# XGBRF Regression
xgbrf_pred = train_and_test(XGBRFRegressor(n_estimators=400))
# kNN
knn_pred_4 = train_and_test(KNeighborsRegressor(n_neighbors = 14))
# Random Forest
rf_pred = train_and_test(RandomForestRegressor(n_estimators=400,random_state=14))
# LGBM Regression
lgbm_pred = train_and_test(LGBMRegressor(boosting_type='gbdt',random_state =94,colsample_bytree=0.9,max_depth=5,subsample=0.9,n_estimators=40))

#%%
# 학습
tree_model = LGBMRegressor(boosting_type='gbdt',random_state =94,colsample_bytree=0.9,max_depth=5,subsample=0.9,n_estimators=40)

tree_model.fit( X_age, y_age )

#%%
# train 데이터의 나이 결측치 예측
train_age = train.loc[train['Age'].isnull(), :]
train_age = age_prediction(train_age)
train_age.describe()
train_age = train_age[['Pclass','Fare', 'Title', 'Has_Cabin',
       'FamilySize','IsAlone']]

y_hat_age = tree_model.predict(train_age)

train_age['Age'] = y_hat_age

nan_index = list(train_age.index)
for index in nan_index :
    train.loc[index, 'Age'] = train_age.loc[index, 'Age']

train['Age'].isnull().sum()

train.head(10)

#%%
# test 데이터의 나이 결측치 예측
test_age = test.loc[test['Age'].isnull(), :]
test_age = age_prediction(test_age)
test_age.describe()
test_age = test_age[['Pclass','Fare', 'Title', 'Has_Cabin',
       'FamilySize','IsAlone']]

y_hat_age = tree_model.predict(test_age)

test_age['Age'] = y_hat_age

nan_index = list(test_age.index)
for index in nan_index :
    test.loc[index, 'Age'] = test_age.loc[index, 'Age']

test['Age'].isnull().sum()

test.head(10)

#%%
# 생존자 예측을 위한 함수 생성
def sur_prediction(temp_for_sur):
    
    # Create new feature FamilySize as a combination of SibSp and Parch
    temp_for_sur['FamilySize'] = temp_for_sur['SibSp'] + temp_for_sur['Parch'] + 1
    
    # mask_temp_for_sur_1 = (temp_for_sur.Age < 10) | (temp_for_sur.Pclass =='1') | (temp_for_sur.Pclass =='2')
    # temp_for_sur['Rich_child'] = mask_temp_for_sur_1.astype(int)
    
    mask_temp_for_sur_2 = (temp_for_sur.Sex == 'male') | (temp_for_sur.Pclass =='2') | (temp_for_sur.Pclass =='3')
    temp_for_sur['Poor_male'] = mask_temp_for_sur_2.astype(int)
    
    mask_temp_for_sur_3 = (temp_for_sur.Sex == 'female') | (temp_for_sur.Pclass =='1') | (temp_for_sur.Pclass =='2')
    temp_for_sur['Rich_female'] = mask_temp_for_sur_3.astype(int)
    
    mask_temp_for_sur_4 = (temp_for_sur.Sex == 'female') | (temp_for_sur.Age <= 10)
    temp_for_sur['Female_child'] = mask_temp_for_sur_4.astype(int)
    
    temp_for_sur['IsAlone'] = 0
    temp_for_sur.loc[temp_for_sur['FamilySize'] == 1, 'IsAlone'] = 1
    
    mask_temp_for_sur_5 = (temp_for_sur.Sex == 'male') | (temp_for_sur.FamilySize >= 5)
    temp_for_sur['Family_male'] = mask_temp_for_sur_5.astype(int)
    
    # Create new feature IsAlone from FamilySize
    temp_for_sur['FamilySize'] = temp_for_sur['FamilySize'].replace({8:0,11:0,6:1,5:2,1:3,7:3,2:4,3:4,4:5})
    # temp_for_sur[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=True)
    """
       FamilySize  Survived
    7           8  0.000000
    8          11  0.000000
    5           6  0.136364
    4           5  0.200000
    0           1  0.303538
    6           7  0.333333
    1           2  0.552795
    2           3  0.578431
    3           4  0.724138
    """
    
    # Create new feature Has_Cabin from Cabin
    temp_for_sur['Has_Cabin'] = temp_for_sur["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    
    # Create new feature FamilySize as a combination of SibSp and Parch
    temp_for_sur['FamilySize'] = temp_for_sur['SibSp'] + temp_for_sur['Parch'] + 1
    
    # Remove all NULLS in the Embarked column
    temp_for_sur['Embarked'] = temp_for_sur['Embarked'].fillna('S')
    
    # Remove all NULLS in the Fare column
    temp_for_sur['Fare'] = temp_for_sur['Fare'].fillna(temp_for_sur['Fare'].median()).astype(int)
    
    # Mapping Sex
    temp_for_sur['Sex'] = temp_for_sur['Sex'].map( {'female': 0, 'male': 1} )
    
    # Mapping Embarked
    temp_for_sur['Embarked'] = temp_for_sur['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    temp_for_sur.loc[ temp_for_sur['Fare'] <= 4, 'Fare'] 		        = 0
    temp_for_sur.loc[(temp_for_sur['Fare'] > 4) & (temp_for_sur['Fare'] <= 8), 'Fare']   = 1
    temp_for_sur.loc[(temp_for_sur['Fare'] > 8) & (temp_for_sur['Fare'] <= 12), 'Fare']   = 2
    temp_for_sur.loc[(temp_for_sur['Fare'] > 12) & (temp_for_sur['Fare'] <= 16), 'Fare']   = 3
    temp_for_sur.loc[(temp_for_sur['Fare'] > 16) & (temp_for_sur['Fare'] <= 20), 'Fare']   = 4
    temp_for_sur.loc[(temp_for_sur['Fare'] > 20) & (temp_for_sur['Fare'] <= 24), 'Fare']   = 5
    temp_for_sur.loc[(temp_for_sur['Fare'] > 24) & (temp_for_sur['Fare'] <= 28), 'Fare']   = 6
    temp_for_sur.loc[(temp_for_sur['Fare'] > 28) & (temp_for_sur['Fare'] <= 32), 'Fare']   = 7
    temp_for_sur.loc[(temp_for_sur['Fare'] > 32) & (temp_for_sur['Fare'] <= 36), 'Fare']   = 8
    temp_for_sur.loc[(temp_for_sur['Fare'] > 36) & (temp_for_sur['Fare'] <= 40), 'Fare']   = 9
    temp_for_sur.loc[(temp_for_sur['Fare'] > 40) & (temp_for_sur['Fare'] <= 44), 'Fare']   = 10
    temp_for_sur.loc[(temp_for_sur['Fare'] > 44) & (temp_for_sur['Fare'] <= 48), 'Fare']   = 11
    temp_for_sur.loc[ temp_for_sur['Fare'] > 48, 'Fare'] 		 = 12
    
    # temp_for_sur[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=True)
    """
        Fare  Survived
    11    11  0.000000
    0      0  0.066667
    2      2  0.225564
    1      1  0.225664
    8      8  0.352941
    9      9  0.388889
    10    10  0.400000
    3      3  0.414414
    6      6  0.433333
    7      7  0.439024
    5      5  0.470588
    4      4  0.500000
    12    12  0.676829
    """
    temp_for_sur['Fare'] = temp_for_sur['Fare'].replace({11:0,0:0,2:1,1:1,8:2,9:3,10:3,3:3,6:3,7:3,5:4,4:5,12:6})
    temp_for_sur['Fare'] = temp_for_sur['Fare'].astype(int)
    
    # Mapping Age
    temp_for_sur.loc[ temp_for_sur['Age']<=10, 'Age'] = 0,
    temp_for_sur.loc[(temp_for_sur['Age']>10)&(temp_for_sur['Age']<=16), 'Age'] = 1,
    temp_for_sur.loc[(temp_for_sur['Age']>16)&(temp_for_sur['Age']<=20), 'Age'] = 2,
    temp_for_sur.loc[(temp_for_sur['Age']>20)&(temp_for_sur['Age']<=26), 'Age'] = 3,
    temp_for_sur.loc[(temp_for_sur['Age']>26)&(temp_for_sur['Age']<=30), 'Age'] = 4,
    temp_for_sur.loc[(temp_for_sur['Age']>30)&(temp_for_sur['Age']<=36), 'Age'] = 5,
    temp_for_sur.loc[(temp_for_sur['Age']>36)&(temp_for_sur['Age']<=40), 'Age'] = 6,
    temp_for_sur.loc[(temp_for_sur['Age']>40)&(temp_for_sur['Age']<=46), 'Age'] = 7,
    temp_for_sur.loc[(temp_for_sur['Age']>46)&(temp_for_sur['Age']<=50), 'Age'] = 8,
    temp_for_sur.loc[(temp_for_sur['Age']>50)&(temp_for_sur['Age']<=60), 'Age'] = 9,
    temp_for_sur.loc[ temp_for_sur['Age']>60, 'Age'] = 10
    
    # temp_for_sur[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=True)
    """
         Age  Survived
    10  10.0  0.227273
    7    7.0  0.326923
    2    2.0  0.341772
    3    3.0  0.342857
    6    6.0  0.377778
    4    4.0  0.400000
    9    9.0  0.404762
    8    8.0  0.470588
    1    1.0  0.472222
    5    5.0  0.472727
    0    0.0  0.593750
    """
        # Define function to extract titles from passenger names
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""
    
    temp_for_sur['Age'] = temp_for_sur['Age'].replace({10:0,7:1,2:1,3:1,6:2,4:3,9:3,8:4,1:4,5:4,0:5})
    
    temp_for_sur['Title'] = temp_for_sur['Name'].apply(get_title)

    # temp_for_sur[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=True)
    """
           Title  Survived
    0       Capt  0.000000
    3        Don  0.000000
    5   Jonkheer  0.000000
    15       Rev  0.000000
    12        Mr  0.156673
    4         Dr  0.428571
    1        Col  0.500000
    7      Major  0.500000
    8     Master  0.575000
    9       Miss  0.697802
    13       Mrs  0.792000
    10      Mlle  1.000000
    11       Mme  1.000000
    2   Countess  1.000000
    14        Ms  1.000000
    6       Lady  1.000000
    16       Sir  1.000000
    """
    temp_for_sur['Title'] = temp_for_sur['Title'].replace(['Capt', 'Don','Jonkheer','Rev'], 0)
    temp_for_sur['Title'] = temp_for_sur['Title'].replace(['Mr'], 1)
    temp_for_sur['Title'] = temp_for_sur['Title'].replace(['Dr'], 2)
    temp_for_sur['Title'] = temp_for_sur['Title'].replace(['Col','Major'], 3)
    temp_for_sur['Title'] = temp_for_sur['Title'].replace(['Master'], 4)
    temp_for_sur['Title'] = temp_for_sur['Title'].replace(['Miss'], 5)
    temp_for_sur['Title'] = temp_for_sur['Title'].replace(['Mrs'], 6)
    temp_for_sur['Title'] = temp_for_sur['Title'].replace(['Mlle','Mme','Countess','Ms','Lady','Sir','Dona'], 7)
    
    
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch']
    temp_for_sur = temp_for_sur.drop(drop_elements, axis = 1)
    temp_for_sur.head()
    
    colormap = plt.cm.viridis
    plt.figure(figsize=(12,12))
    plt.title('Pearson Correlation of Features', y=1, size=10)
    sns.heatmap(temp_for_sur.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
    
    return temp_for_sur

#%%
train = sur_prediction(train)
test = sur_prediction(test)

#%%
# Trainingdataset
train.describe()
train.columns

X = train[ [ 'Pclass', 'Sex', 'Age', 'Fare',
       'Poor_male', 'Rich_female',
       'Family_male', 'Has_Cabin', 'Title'] ]  # 독립변수
y = train['Survived']  # 종속변수

real = test[ ['Pclass', 'Sex', 'Age', 'Fare',
       'Poor_male', 'Rich_female',
       'Family_male', 'Has_Cabin', 'Title'] ]  # 독립변수

#%%

# train data 와 test data로 구분
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10) 

print('train data 개수: ', X_train.shape)
print('test data 개수: ', X_test.shape)
print('\n')

from xgboost import XGBRFClassifier
from lightgbm import LGBMClassifier
from  sklearn.ensemble   import  RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

def train_and_test_survived(model):
    
    model.fit(X_train, y_train)
    y_hat = model.predict(X_train)

    tn, fp, fn, tp = metrics.confusion_matrix( y_train, y_hat ).ravel()
    f1_report = metrics.classification_report( y_train, y_hat )
    print( f1_report )
    
    accuracy = accuracy_score( y_train, y_hat)
    print(accuracy)

    
    return y_hat


# SVM
svm_pred = train_and_test_survived(svm.SVC(kernel='rbf', C=100, gamma=0.01))
#kNN
knn_pred_4 = train_and_test_survived(KNeighborsClassifier(n_neighbors = 6))
# Random Forest
rf_pred = train_and_test_survived(RandomForestClassifier(n_estimators=400, random_state =14))
# XGBRF Classifier
xgbrf_pred = train_and_test_survived(XGBRFClassifier(n_estimators=100))
# LGBM Classifier
lgbm_pred = train_and_test_survived(LGBMClassifier(boosting_type='gbdt',random_state =90,colsample_bytree=0.9,max_depth=5,subsample=0.9,n_estimators=40))

#%%
from xgboost import XGBRFClassifier
from lightgbm import LGBMClassifier
from  sklearn.ensemble   import  RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
model = RandomForestClassifier(n_estimators=400, random_state =14)
model.fit( X, y )

# 테스트 데이터로 예측을 한다.

y_hat = model.predict( real )

results = model.predict( real )
results = pd.Series(results,name="Survived")
submission = pd.concat([pd.Series(range(892,1310),name = "PassengerId"),results],axis = 1)
submission.to_csv("ttest.csv",index=False)
