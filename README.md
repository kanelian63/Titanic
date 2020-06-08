# Goal of Project
Predict Kaggle Titanic Survival by Machine Learning. In particular, it solves the problem of missing values by predicting columns with many missing values by machine learning. https://www.kaggle.com/c/titanic

# Details
In general, it is advantageous to delete a column with many missing values as training data of the model. However, I tried to solve the missing value problem by predicting the missing value by machine learning and filling in the value.

# Heat Map
![Figure_2](https://user-images.githubusercontent.com/59387983/83992613-db1ed280-a98b-11ea-966c-149fa4d6abb9.png)

Age을 예측하기 위해 Age와 어떤 변수와 상관관계가 있는지 알아보기 위해 나타낸 Heat Map

![fssfsf](https://user-images.githubusercontent.com/59387983/83993144-92681900-a98d-11ea-8289-80bdd422bf9a.png)

예측한 Age를 Train, Test Dataset에 추가하고, 어떤 변수가 생존과 상관관계가 있는지 알아보기 위해 Heat Map 출력

# Results
SVM

![XGBRF](https://user-images.githubusercontent.com/59387983/83993464-7c0e8d00-a98e-11ea-8930-01983816a0c9.PNG)

KNN

![LGBM](https://user-images.githubusercontent.com/59387983/83993466-7ca72380-a98e-11ea-8846-2d4da5bd74b7.PNG)

RF

![svm](https://user-images.githubusercontent.com/59387983/83993470-7ca72380-a98e-11ea-873c-4af332dd5c71.PNG)

XGBRF

![KNN](https://user-images.githubusercontent.com/59387983/83993471-7d3fba00-a98e-11ea-88f3-2cc978fbba19.PNG)

LGBM

![RF](https://user-images.githubusercontent.com/59387983/83993472-7dd85080-a98e-11ea-952c-6b429cccff2c.PNG)

일반적으로 LGBM의 성능이 가장 높게 나온다.

# How to run
All source code is in Titanic.py and you can run it as an available program.

# Results
The results were not good. It was confirmed that applying the predicted value to the predictive model was not efficient.

![asda](https://user-images.githubusercontent.com/59387983/83747188-3e161e00-a69b-11ea-9e1d-446b2c123268.PNG)
