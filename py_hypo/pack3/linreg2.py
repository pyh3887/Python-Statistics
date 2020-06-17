# 방법4 : stats.linergress 사용

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

score_iq = pd.read_csv('../testdata/score_iq.csv')
print(score_iq.info())
print(score_iq.head())

# iq와 score 간에 연관 관계를 구한다.
x = score_iq.iq # 독립 변수
y = score_iq.score # 독립 변수

# 상관 계수
# print(np.corrcoef(x, y))
# print(score_iq.corr())
# plt.scatter(x, y)
# plt.show()

# 두 변수 간 인과 관계가 있어 보이므로 회귀 분석을 수행
# LinearRegression(), ols()
model = stats.linregress(x, y)
print(model)    # LinregressResult(slope=0.6514309527270075, intercept=-2.8564471221974657, rvalue=0.8822203446134699, pvalue=2.8476895206683644e-50, stderr=0.028577934409305443)
# newx = 145   # iq
# yhat = 0.6514309527270075 * x + -2.8564471221974657
# print('yhat :', yhat)
print(model.slope)          # 기울기
print(model.intercept)      # 절편
print(model.pvalue)         # pvalue가 0.5 이상이면 신뢰할 수 있는 모델이라고 인정한다. / pvalue가 0.5보다 작으면 독립 변수로 의미가 있다.
