# 분류모델에서 과적합 방지를 위한 조치
# 학습/평가 데이터로 분리 후 모델 작성 : O
# K-fold를 이용해 모델 학습시 검증 작업 함게 실시 : O
# PCA 를 통한 feature 차원 축소
# 학습 조기 종료

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
print(iris.keys())

train_data = iris.data
train_label = iris.target
print(train_data[:1])
print(train_label[:2])


# 분류 모델
dt_clf = DecisionTreeClassifier() # 다른 분류 모델 적용 가능

print(dt_clf)

dt_clf.fit(train_data, train_label)
pred = dt_clf.predict(train_data)
print('예측값 : ' , pred)
print('실제값 : ' , train_label)
print('분류 정확도 ',accuracy_score(train_label, pred))

# 정확도 100% 과적합 발생 주의
# 진짜 시험문제와 모의 시험문제과 가은거와 같다.

print('과적합 방지 ------')
from sklearn.model_selection import train_test_split


x_train, x_test, y_train,y_test = train_test_split(iris.data, iris.target,test_size=0.3, random_state=121)
dt_clf.fit(x_train,y_train) # train
pred2 = dt_clf.predict(x_test)
print('예측값 : ' , pred2[:5])
print('실제값 : ' , y_test[:5])
print('분류 정확도 ',accuracy_score(y_test, pred2)) # 분류 정확도: 0.95 <= 포용성


print('\n 과적합 방지2:위방법도 과적합 발생 가능 k겹 교차검증')
# 진짜 시험 문제를 풀기 전에 모의 시험문제를 여러번(k번)풀어보는 것과 같다.
# train data 학습시 편중을 방지하기위해 train data를 k번 만큼 쪼개서 학습 모델 생성  
from sklearn.model_selection import KFold
import numpy as np
feature = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=123) # 발생 난수 고정 : random_state

kfold = KFold(n_splits=5) # KFold 객체 생성
print(kfold) # 기본 교차검증 스플릿 5개

cv_acc = []
print('iris shape : ' , feature.shape) # (150, 4)
# 전체 행 수 가 150, 학습 데이터: 4/5 (120). 검증 데이터 1/5 (30)로 분할해서 모델 학습

n_iter = 0 

for train_index, test_index in kfold.split(feature):
    print(train_index, test_index)
    xtrain, xtest = feature[train_index], feature[test_index]
    ytrain, ytest = label[train_index], label[test_index]
    
    # 학습 및 옟ㄱ
    dt_clf.fit(xtrain, ytrain)
    pred = dt_clf.predict(xtest)
    n_iter +=1
    # 반복할 때마다 정확도 측정
    acc = np.round(accuracy_score(ytest, pred), 3) # 소수 3째 자리까지 표시
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]
    print('반복수:{}, 교차검증 정확도:{}, 학습데이터크기:{}, 검증데이터 크기:{}'.format(n_iter,acc,train_size,test_size))
    
    cv_acc.append(acc)
print('평균 검증 정확도 : ' , np.mean(cv_acc))


print('과적합 방지2 추가 설명 : 불균형한 분포도를 가진 label 집합을 위한 k-fold 교차 검증')
# 불균형한 분포도 : 특정 레이블 값이 특이하게 많거나 적어서 분포가 왜곡되는 데이터 집합

# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
feature = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=123) # 발생 난수 고정 : random_state

sfold = KFold(n_splits=5) # KFold 객체 생성
print(kfold) # 기본 교차검증 스플릿 5개

cv_acc = []
print('iris shape : ' , feature.shape) # (150, 4)
# 전체 행 수 가 150, 학습 데이터: 4/5 (120). 검증 데이터 1/5 (30)로 분할해서 모델 학습

n_iter = 0 
# Straitfield(Fold는 split(feature, label) 해 준다.
for train_index, test_index in sfold.split(feature, label):
    # print(train_index, test_index)
    xtrain, xtest = feature[train_index], feature[test_index]
    ytrain, ytest = label[train_index], label[test_index]
    
    # 학습 및 옟ㄱ
    dt_clf.fit(xtrain, ytrain)
    pred = dt_clf.predict(xtest)
    n_iter +=1
    # 반복할 때마다 정확도 측정
    acc = np.round(accuracy_score(ytest, pred), 3) # 소수 3째 자리까지 표시
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]
    print('반복수:{}, 교차검증 정확도:{}, 학습데이터크기:{}, 검증데이터 크기:{}'.format(n_iter,acc,train_size,test_size))
    
    cv_acc.append(acc)
print('Stratifield FOle평균 검증 정확도 : ' , np.mean(cv_acc)) # k 번 교차 검증 결과 정확도 평균

print('\n 과적합 방지 2 추가 설명2 : 교차검증을 지원하는 함수 사용')
from sklearn.model_selection import cross_val_score
data = iris.data
label = iris.target

score = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=5)
print('교차검증별 정확도 : ' , np.round(score,3))
print('평균 검증 정확도 : ' , np.round(np.mean(score),3))


print('\n 과적합 방지3 모델 생성시 최적의 속성값  hyper parameter 를 찾아 모델 생성')
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}

grid_dtree = GridSearchCV(dt_clf, param_grid=parameters, cv=5, refit=True)
grid_dtree.fit(x_train, y_train) # 내부적으로 복수 내부 모형이 생성됨. 이를 모두 실행시켜 hyper parameter(최적의속성값)을 찾아냄



import pandas as pd
score_df = pd.DataFrame(grid_dtree.cv_results_)
print(score_df) # rank_test_core가 1인 경우가 최적의 속성값 ( hyper parameter ) 이다

print('hyper parameter : ' , grid_dtree.best_params_)
print('최고 정확도 : ' , grid_dtree.best_score_)

hyper_dt_clf = grid_dtree.best_estimator_ # 최적의 속성값으로 모델 생성
print(hyper_dt_clf)
hyper_pred = hyper_dt_clf.predict(x_test)

print('hyper_pred 정확도 ' , hyper_pred)
print('hyper dt clf 정확도 : ' , accuracy_score(y_test,hyper_pred))