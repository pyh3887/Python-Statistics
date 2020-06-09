# 세 개 이상의 모집단에 대한 가설검정 – 분산분석
# ‘분산분석’이라는 용어는 분산이 발생한 과정을 분석하여 요인에 의한 분산과 요인을 통해 나누어진 각 집단 내의 분산으로 나누고 요인
# 에 의한 분산이 의미 있는 크기를 크기를 가지는지를 검정하는 것을 의미한다.
# 세 집단 이상의 평균비교에서는 독립인 두 집단의 평균 비교를 반복하여 실시할 경우에 제1종 오류가 증가하게 되어 문제가 발생한다.
# 이를 해결하기 위해 Fisher가 개발한 분산분석(ANOVA, ANalysis Of Variance)을 이용하게 된다.

# * 서로 독립인 세 집단의 평균 차이 검정
# 실습) 세 가지 교육방법을 적용하여 1개월 동안 교육받은 교육생 80명을 대상으로 실기시험을 실시. three_sample.csv'

#일원분산분석( one-way anova) - 집단을 구분하는 요인이 1개

#귀무 : 세가지 교육방법(3집단)에 교육방법에 따른 실기시험에 평균에 차이가 없다
#대립 : 세가지 교육방법(3집단)에 교육방법에 따른 실기시험에 평균에 차이가 있다.

import pandas as pd 
import scipy.stats as stats
from statsmodels.formula.api import ols

data = pd.read_csv('../testdata/three_sample.csv')
print(data.head(3), ' ', len(data))
print(data.describe())

import matplotlib.pyplot as plt
#plt.hist(data.score)
#plt.boxplot(data.score)
#plt.show()
data = data.query('score <= 100') # 이상치(outlier)제거
print(data.describe())

#정규성 확인 
print('정규성 만족 여부 :' ,stats.shapiro(data.score)[1]) #정규성 만족 여부 : 0.2986918091773987 > 0.05 ok







