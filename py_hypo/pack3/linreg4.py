# iris dataset으로 선형 회귀
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')
print(iris.head(2))
print(iris.corr())

# 단순 선형 회귀 모델 작성 sepal_length, sepal_width    : -0.117570(r)
result = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit()
# print(result.summary())
print('결정 계수 :', result.rsquared)   # 0.013822654141080748
print('p-value :', result.pvalues)   # p-value : 모델 = 1.518983e-01

print()

# 단순 선형 회귀 모델 작성 sepal_length, petal_length    : 0.871754(r)
result2 = smf.ols(formula='sepal_length ~ petal_length', data=iris).fit()
# print(result.summary())
print('결정 계수 :', result2.rsquared)   # 0.7599546457725151
print('p-value :', result2.pvalues)   # p-value : 모델 = 1.038667e-47
print('실제 값 :', iris.sepal_length[0], ', 예측 값 :', result2.predict()[0])
# 실제 값 : 5.1 , 예측 값 : 4.879094603339241

print()

# 새로운 데이터로 예측(petal_length)로 sepal_length를 예측 한다.
new_data = pd.DataFrame({'petal_length':[1.4, 2.4, 0.4]})
y_pred = result2.predict(new_data)
print(y_pred)

print() 

print('-----------------------------------------------------------------------')
# 다중 선형 회귀 모델 작성
# result3 = smf.ols(formula='sepal_length ~ petal_length + sepal_width + petal_width', data=iris).fit()
# print(result3.summary())
col_selected = "+".join(iris.columns.difference(['sepal_length', 'species']))   # 제외할 칼럼 2개를 입력한다. 
print(col_selected)
formula = 'sepal_length ~ ' + col_selected
result3 = smf.ols(formula=formula, data=iris).fit()
print(result3.summary())