import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


# tv,radio,newspaper 간의 상관관계를 파악하시오. 

#그리고 이들의 관계를 heatmap 그래프로 표현하시오. 



data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/Advertising.csv')

del data['no']
del data['sales']

print(data)
#표준편차알아보기

print(np.std(data.tv),'\n') #85.63933175679271 
print(np.std(data.radio),'\n') #0.8580
print(np.std(data.newspaper),'\n') #0.827

# 시각화 
plt.hist([np.std(data.tv),np.std(data.radio),np.std(data.newspaper)])
plt.show()



#공분산 출력 
print(np.cov(data.tv,data.radio),'\n') #numpy는 np.cov(변수1,변수2)
print(np.cov(data.newspaper,data.radio),'\n')
print(data.cov(),'\n')
print()

#상관계수 출력
print(np.corrcoef(data.tv,data.radio),'\n')
print(np.corrcoef(data.tv,data.newspaper),'\n')

print(data.corr(method='pearson'),'\n') # 변수가 등간/ 비율 척도 일 때. 정규성을 따르는 경우
#print(data.corr(method='spearman'),'\n') # 변수가 서열 척도 일 때. 정규성을 따르지 않는 경우
#print(data.corr(method='kendall'),'\n') # spearman과 유사

# 시각화 (heat map (색으로 표현))

import seaborn as sns
print(data)
plt.rc('font', family='malgun gothic')
sns.heatmap(data.corr())
plt.show()
 
