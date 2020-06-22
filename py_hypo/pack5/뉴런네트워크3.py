# 단층 Perceptron은 XOR 분류를 하지 못함. 층(Layer)를 늘리면 가능. 이를 MLP라고 한다.

import numpy as np
from sklearn.linear_model import Perceptron

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
label = np.array([0,1,1,0])
print(feature)

ml = Perceptron(max_iter=1000).fit(feature,label)
print(ml.predict(feature))

print('----------------ㄴ')

# 다층 신경망(MLP) 사용

from sklearn.neural_network import MLPClassifier
#ml2 = MLPClassifier(hidden_layer_sizes=1).fit(feature,label)
#ml2 = MLPClassifier(hidden_layer_sizes=1000,verbose=1).fit(feature,label) #반복 학습을 시킬때마다 최적의 모델이 만들어진다 
ml2 = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter= 100,
                    random_state=1,verbose=1,learning_rate_init = 0.01).fit(feature,label) #반복 학습을 시킬때마다 최적의 모델이 만들어진다
#print(ml2)

print(ml2.predict(feature))