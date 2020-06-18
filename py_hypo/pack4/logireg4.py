

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import numpy as np 


# 분류 연습용 샘플 데이터 작성

x , y = make_classification(n_samples= 16, n_features = 2, n_informative = 2, n_redundant = 0, random_state = 0)

print(x) #[[ 2.03418291 -0.38437236] 
print(y) # [0 1 0 1 1 0 0 0 1 0 1 0 1 1 0 1]

model = LogisticRegression().fit(x,y)
y_hat = model.predict(x)
print('y_hat :' , y_hat)

f_value = model.decision_function(x) # 결정 함수
print('f_value :' , f_value) 

print()

df = pd.DataFrame(np.vstack([f_value, y_hat]).T, columns =['f','yhat'])
print(df) 


# roc 커브 

from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y,y_hat,labels =[1,0]))

recall = 7 / (7 + 1) # TP/ (TP + FN) -TPR
fallout =1 / (1 + 7) # FP / (FP +TN) -FPR

print('recall:', recall)
print('recall', fallout)

from sklearn.metrics import roc_curve
fpr , tpr ,thesold = roc_curve(y,model.decision_function(x))
print('fpr:', fpr)
print('tpr:', tpr)
print( 'threshold' , thesold) # 재현을 높이귀 위한 판단 기준


import matplotlib.pyplot as plt 
plt.rc('font', family = 'malun gohic')
plt.plot(fpr,tpr,'o-', label = 'Logistic Regression')
plt.plot([0,1],[0,1] , 'k--' , 'random guess')
plt.plot([fallout],[recall],'ro' , ms= 10)
plt.xlabel('위양성율(fpr:fallout) ')
plt.xlabel('재현율(tpr:recall)')
plt.show()
