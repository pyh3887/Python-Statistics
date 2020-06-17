# 분류 결과가 두가지 이상인 경우 다항분류 모델을 사용 LogisticRegression

from sklearn import datasets
from sklearn.model_selection._split import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(iris.keys()) 
# print(iris.DESCR)
# print(iris.data,iris.data.shape) # feature (독립 변수)
# print(iris.target, iris.target.shape) # label(class,종속변수) (150,)

#x = iris.data

print()
# 참고: 확률에 의해 꽃 종류가 결정되는 결정 간격 확인 

# 꽃잎의 넓이에 따라 꽃의 종류를 나누고 있으며 , setosa와 versicolor를 묶고 , verginica 와 나누는데 , 
# 현재 기준값인 1.7 이상일시(확률 0.5) verginica로 채택하며 , 1.7 아래시에는 versinica 가 아니다를 채택한다. 
log_reg = LogisticRegression() # 분석모델 객체 생성

print(log_reg)
x = iris['data'][:,3:] # petal.width 꽃잎 넓이 만 작업에 참여
y = (iris['target']== 2).astype(np.int) #setosa - versicolor 1 / verginica 1 0,1로 target 을 나눔
#print(x)
log_reg.fit(x,y) 

x_new = np.linspace(0, 3, 1000).reshape(-1,1) # -1 의 의미 행은 알아서
print(x_new.shape) # (1000,1) 2차원배열
y_proba = log_reg.predict_proba(x_new)
print(y_proba) # 확률 값으로 출력 [9.99250016e-01 7.49984089e-04]

# plt.plot(x_new,y_proba[:,1],'-r', label='virginicar')
# plt.plot(x_new,y_proba[:,0],'b--', label='not virginicar')
# plt.legend()
# plt.show()
print(log_reg.predict([[1.5],[1.7]])) # not virginicar , virginicar
print(log_reg.predict([[2.5],[0.7]])) # virginicar , not virginicar
print(log_reg.predict_proba([[2.5],[0.7]])) #[[0.02563061 0.97436939]
# 결론적으로 LogisticRegression() 출력 값이 두 개 이상인 경우에 있어 확률값이 0.5 이상인 요소의 index 가 출력  

#LogisticRegression()은 다중 클래스(label)를 지원하도록 일반화되어있다. softmax 함수를 사용

print('-----------------------------')

x = iris.data[:,[2,3]]  # 꽃받침 x petal.length, petal.width (꽃의길이,꽃의넓이) 칼럼으로 꽃의 종류를 3가지로 분류
y = iris.target

# print(x[:3])
# print(y[:3],' ' ,set(y))


#train / test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3 , random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # x 꽃잎의 넓이 ,길이  y 0,1,2 종속변수

#스케일링(데이터 크기 표준화) 단위를 같게해야함
# 방법1 정규화(모든 값들을 0~1사이의 값)
# 방법2 표준화 (평균의 표준편차 내로 값을 맞춤) : 전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정이다.
#스케일링의 효과 -> 정확성이 높아짐
print(x_train[:3])
sc = StandardScaler() # 평균이 0 표준편차가 1이 되도록 변환,
sc.fit(x_train)
sc.fit(x_test)

x_train = sc.transform(x_train) #데이터 표준화
x_test = sc.transform(x_test) #데이터 표준화
print(x_train[:3])  

print('분류 모델 사용 ---------------')

ml = LogisticRegression(C=0.1,random_state=0) # C= 모델에 패널티를 적용(오버피팅 관련) , 모델 정확도 조정
print(ml)
result = ml.fit(x_train,y_train) # train data 로 모델 학습 << 학습된 데이터로  [최적의 회귀모델을 찾는다.]

#print(result)

# 모델 학습 후 객체를 저장

import pickle
fileName = 'final_model.sav'
#pickle.dump(ml,open(fileName,'wb'))
ml = pickle.load(open(fileName,'rb'))

#-----------------

#분류 예측
y_pred = ml.predict(x_test) 
print('예측값:' , y_pred) #예측값과 실제값이 거의 일치함.
print('실제값:' , y_test) 
print('분류 정확도 :' ,accuracy_score(y_test,y_pred)) # 정확도 97.7 %

# confusion_matrix 
con_mat = pd.crosstab(y_test,y_pred,rownames=['예측값'],colnames=['관측값']) 
print(con_mat)
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2])/len(y_test)) #0.9777777777777777
print(ml.score(x_test,y_test)) # 0.9777777777777777

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from matplotlib import font_manager, rc



font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()

plt.rc('font', family=font_name)      #그래프에서 한글깨짐 방지용



def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02, title=''):

    markers = ('s', 'x', 'o', '^', 'v')  # 점표시 모양 5개 정의

    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    #print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])

    

    # decision surface 그리기

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    

    # xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의 

    # predict()의 안자로 입력하여 계산된 예측값을 Z로 둔다.

    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)

    Z = Z.reshape(xx.shape) #Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.

    

    # X를 xx, yy가 축인 그래프상에 cmap을 이용해 등고선을 그림

    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())

    

    X_test = X[test_idx, :]

    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl)

        

    if test_idx:

        X_test = X[test_idx, :]

        plt.scatter(X_test[:, 0], X_test[:, 1], c='', linewidth=1, marker='o', s=80, label='testset')

    

    plt.xlabel('표준화된 꽃잎 길이')

    plt.ylabel('표준화된 꽃잎 너비')

    plt.legend(loc=2)

    plt.title(title)

    plt.show()



x_combined_std = np.vstack((x_train, x_test))

y_combined = np.hstack((y_train, y_test))

plot_decision_region(X=x_combined_std, y=y_combined, classifier=ml, 

                    test_idx=range(105, 150), title='scikit-learn제공')  


# 새로운 값으로 예측
import numpy as np 
new_data = np.array([[5.1,2.4],[0.3,0.3],[1.4,3.4]])
new_pred = ml.predict(new_data)
print('예측 결과 : ', new_pred) # 예측 결과 :  [2 1 2] 




""" Softmax 구현 """
import numpy as np

def softmax(a) :
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

a = np.array([0.3, 2.9, 4.0])

print (softmax(a)) # softmax 결과값 출력
print (sum(softmax(a))) # softmax 결과값들의 합은 1이 된다.




       