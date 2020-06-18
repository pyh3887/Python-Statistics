# svm 으로 xor 분류
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import svm , metrics
from sklearn.linear_model._logistic import LogisticRegression

xor_data = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]    
    
    ]

xor_df = pd.DataFrame(xor_data)
print(xor_df)

feature = np.array(xor_df.iloc[:,0:2])
label = np.array(xor_df.iloc[:,2])
print(feature)

#model = LogisticRegression()
model = svm.SVC()
model.fit(feature,label)

pred = model.predict(feature) # 예측값
print('pred :', pred)

# 분류 리포트
acc = metrics.accuracy_score(label,pred)
print('분류 정확도 :' , acc)
ac_report = metrics.classification_report(label,pred)
print(ac_report)  # 정밀도 precision    recall

