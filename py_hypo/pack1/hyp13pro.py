# 비율검정 : 집단의 비율이 어떤 특정한 값과 유의한지 검정 
import numpy as np 
from statsmodels.stats.proportion import proportions_ztest

# one - sample
# A교육센터에는 100명 중에 45명이 흡연을 한다. 국가 통계를 확인해 보니 국민 흡연율은 35%로 알려져 있다.
# 이때 비율이 다른지 검정 
# 귀무 : A 교육센터의 흡연율과 국민 흡연율의 비율은 같다. 
# 대립 : A 교육센터의 흡연율과 국민 흡연율의 비율은 같지 않다.

count = np.array([45])
nobs = np.array([100])

val = 0.35

z, p = proportions_ztest(count=count, nobs=nobs, value = val)
print(z)
print(p) # 0.0444 <0.05 귀무기각 대립가설 채택 / 비율은 같지않다

# two sample 
# a 교육센터 300 명중 100명이 햄버거를 먹었고 , b 교육센터 400명중 170명이 햄버거를 먹었다.
# 두 집단의 햄버거를 먹은 비율의 동일여부를 검정하시오. 
# 귀무 : 비율은 같다. 
# 대립 : 비율은 같지 않다.

count = np.array([100,170])
nobs = np.array([300,400])

z, p = proportions_ztest(count=count, nobs=nobs, value = 0)
print(z)
print(p) # 0.0444 <0.05 귀무기각 대립가설 채택 / 비율은 같지않다
