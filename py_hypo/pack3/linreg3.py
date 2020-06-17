# 사용 빈도가 높은 회귀 분석 모델 ols() - 가장 기본적인 결정론적 선형 회귀 방법
import pandas as pd

df = pd.read_csv("../testdata/drinking_water.csv")
print(df.head())
print(df.corr())

import statsmodels.formula.api as smf

model = smf.ols(formula='만족도 ~ 적절성', data=df).fit()     # r style의 모델
# print(model.summary())      # R-squared = 결정 계수 / Prob = pvalue / P>|t| = pvalue
print('\n회귀 계수 :', model.params)
print('\n결정 계수(설명력) :', model.rsquared)
print('\np 값 :', model.pvalues) # p 값 = 1.454388e-09, 적절성 모델 전체 = 2.235345e-52
# print('\n예측 값 :', model.predict())
print('\n', df.만족도[0], model.predict()[0]) # 실제 값 = 3, 예측 값 = 3.7359630488589186

# 시각화
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(df.적절성, df.만족도)
slope, intercept = np.polyfit(df.적절성, df.만족도, 1)   # 1차원
print('slope, intercept :', slope, intercept)
plt.plot(df.적절성, slope * df.적절성 + intercept, 'b')      # y = wx + b / 파란색 선을 긋는다.
plt.show()

print('--------------------------------')
# 다중 선형 회귀
model2 = smf.ols(formula='만족도 ~ 적절성 + 친밀도', data=df).fit()     # r style의 모델
print(model2.summary())