# 비예보

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics


df = pd.read_csv('../testdata/weather.csv')
print(df.head(3))
print(df.info())
x = df[['MinTemp','MaxTemp','Rainfall','Cloud']]
print(df['RainTomorrow'].unique())
# label = df['RainTomorrow].apply(lambda x:1 if x == 'Yes' else 0)
label = df['RainTomorrow'].map({'Yes':1 , 'No':0})
print(label[:5])

# train/ test (7:3) : overfitting 방지 
train_x , test_x , train_y , test_y = train_test_split(x,label,test_size = 0.3 , random_state = 0)
print(len(train_x), ' ' , len(test_x))



# predict
gmodel = GaussianNB()
gmodel.fit(train_x,train_y)
pred = gmodel.predict(test_x)
acc = sum(test_y == pred) / len(pred)
print('정확도 :',acc)
print('정확도 :',accuracy_score(test_y,pred))

# k-fold (교차 검증) - 모델 학습시 입력자료를 k겹으로 나누어 학습과 검증을 함께하는 방법
from sklearn import model_selection
#cross_val = model_selection.cross_val_score(gmodel, x, label, cv = 5) # 원본 x ,y 를 사용
cross_val = model_selection.cross_val_score(gmodel,train_x, train_y, cv = 5) # 원본 x ,y 를 사용
print(cross_val)

print('새로운 자료로 예측')
print(x.head(3))
import numpy as np
# MinTemp MaxTemp RainFall cloud
my_weather = np.array([[14.0,26.9,3.6,3],[2.0,11.9,9.6,30],[19.0,30.9,82.6,80]])
print(gmodel.predict(my_weather))




