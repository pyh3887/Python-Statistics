'''
LOgistic Regression : 지도학습 중 분류모델 (이항분류)
독립변수(연속형) , 종속변수(범주형)
Logit 변환 (odds ratio 의 결과에 log를 씌어 0~1 사이의 확률값을 반환)

sigmoid  function  1/ (1+e **-x)

'''
import math
import numpy as np

def sigmoidFunc(x):
    return 1 / (1+ math.exp(-x))

'''
print(sigmoidFunc(1))
print(sigmoidFunc(3))
print(sigmoidFunc(-2))
print(sigmoidFunc(0.2))
print(sigmoidFunc(0.8))
print(np.around(sigmoidFunc(-0.2)))
print(np.around(sigmoidFunc(0.8)))
print(np.rint(sigmoidFunc(0.8)))
'''
print('car '*7)

import statsmodels.api as sm
import statsmodels.formula.api as smf

# 자동차 데이터로 분류연습  ( 연비와 마력수로 변속기 분류)
mtcars = sm.datasets.get_rdataset('mtcars').data
print(mtcars.head(2))
mtcar = mtcars.loc[:,['mpg','hp','am']] # am :  변속기 종류(수동, 자동)
print(mtcar.head(2))
print(mtcar['am'].unique())  #  [1 0]


# 연습 1 :  logit()
formula = 'am ~ hp +mpg'
result = smf.logit(formula = formula, data = mtcar).fit()
print(result)
print(result.summary())

pred = result.predict(mtcar[:5])

print('예측값 : ',np.around(pred))
print('실제값 : ',mtcar['am'][:5])

# confusion matrix
conf_tab = result.pred_table()
print(conf_tab)
print('분류 정확도 (accuracy) : ',(16+10) /len(mtcar))
print('분류 정확도 (accuracy) : ',(conf_tab[0][0]+conf_tab[1][1]) /len(mtcar))

from sklearn.metrics import accuracy_score

pred2 = result.predict(mtcar)
print('분류 정확도 (accuracy) : ',accuracy_score(mtcar['am'],np.around(pred2)))

''''
분류 정확도 (accuracy) :  0.8125
분류 정확도 (accuracy) :  0.8125
분류 정확도 (accuracy) :  0.8125
'''

#  연습2 : glm()
result2 = smf.glm(formula = formula, data = mtcar, family= sm.families.Binomial()).fit()
print(result2)
print(result2.summary())
glm_pred = result2.predict(mtcar[:5])
print('glm_pred :' , glm_pred)
print('glm_pred :' , np.around(glm_pred))

glm_pred2 = result2.predict(mtcar)
print('분류 정확도 : ', accuracy_score(mtcar['am'], np.around(glm_pred2)))#  분류 정확도 검사 : 0.8125


# 머신 러닝의 포용성  : 머신 러닝은 수학이다 아니라 추론이다... 확률 , 가능성 

# 새로운 값으로 예측
#newdf = mtcar.iloc[:2]
newdf = mtcar.iloc[:2].copy() # 기존 데이터를 일부 추출해 새 객체 생성 후 분류 작업 
print(newdf) 
newdf['mpg'] = [10,30]
newdf['hp'] = [100,120]

print(newdf)

new_pred = result2.predict(newdf)
print('새로운 데이터에 대한 분류 결과 : ', np.around(new_pred)) # 10 , 100 은 0으로 30,120 은 1.0으로 분류
print('새로운 데이터에 대한 분류 결과 : ', np.rint(new_pred))
print('---')

import pandas as pd
newdf2 = pd.DataFrame({'mpg':[10,35],'hp':[100,125]})
new_pred2 = result2.predict(newdf2)
print(newdf2)
print('새로운 데이터에 대한 분류 결과 : ', np.around(new_pred2)) # 10 , 100 은 0으로 30,120 은 1.0으로 분류




 










