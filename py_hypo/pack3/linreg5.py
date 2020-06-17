# mtcars dataset 으로 선형회귀 분석

#귀납적 /연약적 추론. 통계학은 귀납적 추론(개별 사례를 모아 일반적인 법칙(모델)을 생성)

import statsmodels.api 
import statsmodels.formula.api as smf
import numpy as np 

import pandas as pd 
import matplotlib.pyplot as plt
plt.rc('font', family = 'malgun gothic')

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data

print(mtcars)  # [32 rows x 11 columns]
print(mtcars.describe())
print(mtcars.info())

#print(mtcars.corr())

print(np.corrcoef(mtcars.hp,mtcars.mpg)) # 마력수 , 연비 상관관계 -0.7761 

plt.scatter(mtcars.hp, mtcars.mpg)
plt.xlabel('마력수')
plt.ylabel('연비')
slope,intercept = np.polyfit(mtcars.hp,mtcars.mpg, 1) #1차원 R의 abline의 효과
print(slope*mtcars.hp + intercept) # wx + b
plt.plot(mtcars.hp, mtcars.hp * slope + intercept, 'r')
plt.show()

print('\n---단순 선형회귀 분석 -----------')
result = smf.ols(formula = 'mpg ~ hp', data=mtcars).fit()
#print(result.conf_int(alpha= 0.05)) # 95%
#print(result.conf_int(alpha= 0.01)) # 99%
print(result.summary().tables[1])
# yhat = -0.0682 * x + 30.0989
print('예측 연비 :' ,-0.0682 * 110 + 30.0989) #예측 연비 : 22.5969
print('예측 연비 :' ,-0.0682 * 50 + 30.0989)  #예측 연비 : 26.6889
print('예측 연비 :' ,-0.0682 * 200 + 30.0989) #예측 연비 : 16.4589
# msg(연비)는 hp (마력수) 값에 -0.0682 배 씩 영향을 받고 있다.
# 마력에 따라 연비는 증감한다. 라고 말할수 있으나 이는 조심스럽다. 일반적으로 독립변수는 복수 ...
# 모델이 제공한 값을 믿고 섣불리 판단하는 것은 곤랂다ㅏ. 의사결정을 위한 참고 자료로 사용해야한닫. 

print('\n--- 다중 선형회귀 분석 (독립변수 복수) ------')
result2 = smf.ols(formula = 'mpg ~ hp + wt', data=mtcars).fit()
print(result2.summary())

print('\n--------추정치 구하기 : 임의의 마력수와 차체무게에 대한 연비 출력-----------')
result3 = smf.ols(formula='mpg ~ wt', data = mtcars).fit() # wt 차체 무게

print(result3.summary())

# 결정계수 : 0.753
# 모델의 p-value : 1.29e -10

# 추정치 (예측값) 출력 
kbs = result3.predict()
print('예측값' , kbs[:2])
print('예측값' , mtcars.mpg[:2])

data = {
    'mpg':mtcars.mpg,
    'mpg_pred':kbs,
    
    }
df = pd.DataFrame(data)
print(df) # 실제 연비와 추정 연비가 대체적으로 비슷한 것을 알 수 있다.

print()
# 이제 새로운 데이터(차체 무게)로 연비를 추정
mtcars.wt = 6 # 차체 무게가 6톤이라면 연비는 ? 
ytn = result3.predict(pd.DataFrame(mtcars.wt))
print('차체 무게가  6톤이라면 연비는?', ytn[0]) # 5.218

mtcars.wt = 0.4 # 차체 무게가 400kg이라면 연비는 ? 
ytn = result3.predict(pd.DataFrame(mtcars.wt))
print('차체 무게가  400kg이라면 연비는?', ytn[0]) # 35.14

#복수 차체무게에 대한 연비 예측 

wt_new = pd.DataFrame({'wt':[6,3,1,0.4,0.3]})
pred_mpgs = result3.predict(wt_new)
print('예쌍 연비 :', np.round(pred_mpgs.values,3)) # [ 5.218 21.252 31.941 35.147 35.682]


















