from sklearn.datasets import load_breast_cancer
import numpy as np


cancer =load_breast_cancer()
print(cancer.keys())

x= cancer['data']
y= cancer['target']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
scaler.fit(x_test)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train[:1])

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), random_state =1)
print(mlp)
mlp.fit(x_train,y_train)

pred = mlp.predict(x_test)
print('예측값: ',pred[:10]) # 예측값:  [0 1 0 1 0 1 1 1 0 1]
print('실제값: ',y_test[:10]) # 실제값:  [0 1 0 1 0 1 1 1 0 1]
print('분류 정확도 : ', mlp.score(x_test,y_test)) # 분류 정확도 :  0.98601

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))



 







