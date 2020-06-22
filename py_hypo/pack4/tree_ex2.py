# RandomForest vs xgboost

import pandas as pd

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

import numpy as np

import xgboost as xgb  # pip install xgboost  
import matplotlib.pyplot as plt


if __name__ == '__main__':

    data = pd.read_csv('glass.csv')

    

    print(data.head(2))

    x = data[['RI', 'Na', 'Mg', 'Al','Si','K','Ca','Ba','Fe']]

    y = data['Type']

    # 테스트 데이터 30%

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

    # 학습 진행

    # model = RandomForestClassifier(n_estimators=100)  # RandomForestClassifier

    model = xgb.XGBClassifier(booster='gbtree',  # XGBClassifier

                    max_depth=4,

                    n_estimators=100) 

   # 속성 - booster: 의사결정 기반 모형(gbtree), 선형 모형(linear)

     #            - max_depth [기본값: 6]: 과적합 방지를 위해서 사용되며 CV를 사용해서 적절한 값이 제시되어야 하고 보통 3-10 사이 값이 적용된다.

    model.fit(x_train, y_train)

    # 예측

    y_pred = model.predict(x_test)

    print('예측값 : ', y_pred[:5])

    print('실제값 : ', np.array(y_test[:5]))

    print('정확도 : ', metrics.accuracy_score(y_test, y_pred))
    
    
    

def plot_feature_importances(model):
    n_features = x.shape[1]
    # 바차트(horizon)
    
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x.columns)
    plt.xlabel("attr importances")
    plt.ylabel("attr")
    plt.ylim(-1, n_features)
    plt.show()
    plt.close()
    
plot_feature_importances(model)



