# 두 집단의 가설검정 – 실습 시 분산을 알지 못하는 것으로 한정하겠다.
# * 서로 독립인 두 집단의 평균 차이 검정(independent samples t-test)
# 남녀의 성적, A반과 B반의 키, 경기도와 충청도의 소득 따위의 서로 독립인 두 집단에서 얻은 표본을 독립표본(two sample)이라고 한다.
# 실습) 남녀 두 집단 간 파이썬 시험의 평균 차이 검정
# Male = [75, 85, 100, 72.5, 86.5]
# female = [63.2, 76, 52, 100, 70]
# 실습) 두 가지 교육방법에 따른 평균시험 점수에 대한 검정 수행 two_sample.csv'

# 귀무 : 남녀 두 집단 간 파이썬 시험의 평균에 차이가 없다
# 귀무 : 남녀 두 집단 간 파이썬 시험의 평균에 차이가 있다

from scipy import stats
import pandas as pd 
from numpy import average

male = [75,85,100,72.5,86.5]
female = [63.2,76,52,100,70]

#two_sample = stats.ttest_ind(male,female)
two_sample = stats.ttest_ind(male,female,equal_var=True) # 등분산성 만족(두 그룹간의 분산이 같다)
print(two_sample)
#Ttest_indResult(statistic=1.233193127514512, pvalue=0.2525076844853278)
sta,pv = two_sample
print('sta :' , sta)
print('pv :' , pv)
print(average(male), ' ' , average(female)) 
# 해석: male 평균이 83.8 female 평균이 72.24
#pvalue: 0.25250 > 0.05 이므로 귀무가설 채택 . 남녀 두집단간 파이썬 시험의 평균에 차이가 없다
