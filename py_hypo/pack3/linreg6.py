# mtcars dataset 으로 선형회귀 분석 : LinearRegression

from sklearn.linear_model import LinearRegression
import statsmodels.api 
import matplotlib.pyplot as plt
mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data

x = mtcars[['hp']].values # matrix로 얻기
y = mtcars[['mpg']].values

#print(x, x.shape)
#print(y, y.shape) 

# 시각화

plt.scatter(x,y)
#plt.xlabel('마력수')
#plt.ylabel('연비')
#plt.show()

fit_model = LinearRegression().fit(x,y) # 기울기(가중치), 절편(추정된 상수 , bias , 편향) 반환

print(fit_model.coef_)
print(fit_model.intercept_)

print('기울기(가중치):', fit_model.coef_[0]) #[-0.06822828]
print('절편(추정된 상수 , bias , 편향):', fit_model.intercept_) #[30.0988]

# 참고 : dataset을 train / test로 분리(7:3) 후 모델 학습 후 모델 평가
pred = fit_model.predict(x) # 사실은 train data로 학습하고 , test data로 예측값 보기 (모델평가)

# 새로운 값으로 연비  추정
new_hp = [[110]]
new_pred = fit_model.predict(new_hp)
print('new_pred : ' , new_pred[[0][0]])



