# 상관분석 : 두 개 이상의 변수 간에 어떤 관계가 있는지 분석하는 것 
# 공분산을 표준화함 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.DataFrame({'id1':(1,2,3,4,5),'id2':(2,3,-1,7,9)})
print(df)
#plt.scatter(df.id1,df.id2)
#plt.show()

print('공분산:\n' ,df.cov()) # 관련 방향성 제시하나 강도 표현은 모호하다
print('\n상관계수\n:',df.corr()) # 공분산의 약점을 해결

print('\n\n상품의 만족도 관계 확인--')
data = pd.read_csv('../testdata/drinking_water.csv')

print(data.head())
print(data.describe())

# 각 요소의 표준편차 출력 

print(np.std(data.친밀도),'\n') #0.9685
print(np.std(data.적절성),'\n') #0.8580
print(np.std(data.만족도),'\n') #0.8271
 
# 시각화 
#plt.hist([np.std(data.친밀도),np.std(data.적절성),np.std(data.만족도)])
#plt.show()

#공분산 출력 
print(np.cov(data.친밀도,data.적절성),'\n') #numpy는 np.cov(변수1,변수2)
print(np.cov(data.친밀도,data.만족도),'\n')
print(data.cov(),'\n')
print()
#상관계수 출력
print(np.corrcoef(data.친밀도,data.적절성),'\n')
print(np.corrcoef(data.친밀도,data.만족도),'\n')

print(data.corr(method='pearson'),'\n') # 변수가 등간/ 비율 척도 일 때. 정규성을 따르는 경우
#print(data.corr(method='spearman'),'\n') # 변수가 서열 척도 일 때. 정규성을 따르지 않는 경우
#print(data.corr(method='kendall'),'\n') # spearman과 유사

# 시각화 (heat map (색으로 표현))

import seaborn as sns
plt.rc('font', family='malgun gothic')
sns.heatmap(data.corr())
plt.show()
 


















