import scipy.stats as stats
import pandas as pd 
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt 
import urllib.request


url = 'https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3.txt'
data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')
print(data)

#그룹별(3개) 네과목 시험점수 차이 검정
# 귀무 : 그룹별 네과목 시험점수는 차이가 없다 
# 대립 : 그룹별 네과목 시험점수의 차이는 있다.

#data2 = pd.read_csv(urllib.request.urlopen(url))
#print(data2)

gr1 = data[data[:,1] == 1, 0]
#print(grl)
gr2 = data[data[:,1] == 2, 0]
gr3 = data[data[:,1] == 3, 0]

print(stats.shapiro(gr1)[1]) #0.3336853086948395  > 0.05  정규성을띔
print(stats.shapiro(gr2)[1]) #0.6561065912246704  > 0.05  정규성을띔
print(stats.shapiro(gr3)[1]) #0.832481324672699   > 0.05  정규성을띔

# 그룹 간 데이터 들의 분포를 시각화 
#plot_data = [gr1,gr2,gr3]
#plt.boxplot(plot_data)
#plt.show()

f_statistic, p_val = stats.f_oneway(gr1,gr2,gr3)
print('일원분산분석 결과 : f_statistic:%f , p_val:%f'%(f_statistic,p_val))
# 일원분산분석 결과 : f_statistic:3.711336 , p_val:0.043589 <0.05 이므로 귀무기각
 
# 그룹별 (3개) 시험점수는 차이가 있다 라는 의견이 통계적으로 유의하다

#일원분산분석 방법2 - Linear Model 을 속성으로 사용 
df = pd.DataFrame(data, columns = ['value','group'])
#print(df)
lmodel = ols('value ~ C(group)', df).fit() # C(그룹칼럼..) : 범주형임을 명시적으로 표시  PR(>F)=p-value 0.043589
print(anova_lm(lmodel))


#이원분산분석 : 집단 구분 요인2
url = 'https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3_2.txt'
data = pd.read_csv(url)
print(data.head(3))
print(data.tail(3))

#귀무  : 관측자와 태아수 그룹에 따라 태아의 머리둘레에 차이가 없다. 
#대립  : 관측자와 태아수 그룹에 따라 태아의 머리둘레에 차이가 있다. 

# 시각화 
plt.rc('font', family = 'malgun gothic')
data.boxplot(column = '머리둘레' , by='태아수' , grid = True)
#plt.show() # 태아의 머리둘레는 차이가 있어 보임 . 관측자와 상호 작용이 있는지 분산분석으로 검정
formula = '머리둘레 ~ C(태아수) + C(관측자수) + C(태아수):C(관측자수)'
lm = ols(formula = formula, data = data).fit() #학습한 결과를가지고 객체를 만들어냄
print(anova_lm(lm)) # 아노바

# C(태아수)          해석 :  1.051039e-27 < 0.05 이므로 머리둘레에 차이가 있다.

# C(관측자수)         해석 : 6.497055e-03 < 0.05 이므로 머리둘레에 차이가 있다. 

# C(태아수):C(관측자수) 해석 : 3.295509e-01 > 0.05 이므로 머리둘레에 차이가 없다.
 
# 결과 : 관측자수와 태아수는 머리둘레에 영향을 미치나 , 관측자수와 태아수에 상호 작용에 의한 영향은 없다.

print()
formula2 = '머리둘레 ~ C(태아수) + C(관측자수)'  
lm2 = ols(formula = formula2, data = data).fit() # 상호작용 X

print(anova_lm(lm2)) 














