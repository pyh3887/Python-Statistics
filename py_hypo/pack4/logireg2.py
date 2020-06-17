#날씨 정보 자료를 이용해 날씨 예보(내일 비 유무)

import pandas as pd 
from sklearn.model_selection._split import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/weather.csv')

print(data.head(2),data.shape) # (366.12)
data2 = pd.DataFrame()
data2 = data.drop(['Date','RainToday'],axis= 1)
data2['RainTomorrow'] = data2['RainTomorrow'].map({'Yes':1,'No':0}) #내일날씨자료의 yes 는 1 로 no 는0 으로 바꿈 

print(data2.head(2), data2.shape)

# RainTomorrow : 종속변수 , 나머지열 : 독립변수

# train / test dataset 으로 분리 : 과적합(overfitting) 방지목적

train , test = train_test_split(data2, test_size = 0.3, random_state = 42) # 데이터 셔플링후 30 % 의 데이터를 뽑음
print(data.shape,train.shape,test.shape)

# 분류 모델 
my_formula = 'RainTomorrow ~ MinTemp + MaxTemp + Rainfall....'
col_select = '+'.join(train.columns.difference(['RainTomorrow']))
my_formula = 'RainTomorrow ~' + col_select
print(my_formula)

# 분류를 위한 학습모델의 생성
#model = smf.glm(formula = my_formula, data = train, family = sm.families.Binomial()).fit() #모델을 fitting 시킬땐 trian
model = smf.logit(formula = my_formula, data = train).fit() #모델을 fitting 시킬땐 trian

#print(model.summary())
#print(model.params())
print('예측값:' , np.rint(model.predict(test)[:5])) # 모델을 예측할때는 test
print('실제값:' , test['RainTomorrow'][:5])

# 분류 정확도 

conf_mat = model.pred_table()
print(conf_mat)
print((conf_mat[0][0]+ conf_mat[0][0]) / len(train))

from sklearn.metrics import accuracy_score
pred = model.predict(test)
print('분류 정확도 : ', accuracy_score(test['RainTomorrow'],np.around(pred)))