# BMI (체질량 지수) 식을 이용해 무작위 자료를 작성 후 분류 모델에 적용
# BMI = 몸무게 / 키 ^2

print(75 / (1.7 * 1.7))

# import random
#  
# random.seed(123)
#  
# def calc_bmi(h,w):
#     bmi = w / (h/100) ** 2 
#     if bmi < 18.5 : return 'thin'
#     if bmi < 23 : return 'normal'
#     return 'fat'
#  
# print(calc_bmi(170,65))
#  
# #bmi data 생성 후 파일로 저장
#  
# fp = open('bmi.csv','w',encoding='utf-8')
# fp.write('height,weight,label\n')
# #무작위로 데이터를 생성
# cnt = {'thin':0,'normal':0,'fat':0}
#  
# for i in range(50000):
#     h = random.randint(150,200) # 150에서 200사이의 키 데이터 랜덤생성
#     w = random.randint(35,100) # 35에서 100사이의 몸무게 데이터 랜덤생성
#     label = calc_bmi(h, w) # 데이터를 넣었을시 결과값 
#     cnt[label] += 1
#     fp.write('{0},{1},{2}\n'.format(h,w,label)) #dict 타입으로 담기
#      
# fp.close()
# print('저장완료', cnt)

#SVM 으로 bmi data 분류 
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 

tbl = pd.read_csv('bmi.csv')
#print(tbl.head(3))

# w , h 에 대해 정규화 하기 

label = tbl['label']
w = tbl['weight'] / 100 # 0 ~ 1 사이로 정규화 
h = tbl['height'] / 200 

wh = pd.concat([w,h],axis = 1)
print(wh.head(3),wh.shape)
print(label[:3],label.shape)

# train / test dataset : 과적합 방지용
data_train,data_test,label_train,label_test = train_test_split(wh,label)
print(data_train.shape,data_test.shape)

# model

#model = svm.SVC().fit(data_train,label_train)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5).fit(data_train,label_train) #일반적으로 k값은 3,5 를 부여한다.
pred = model.predict(data_test)
print('실제값 : ' ,label_test[:3])
print('예측값 : ' ,pred[:3])

# k 겸 교차검증 : 과적합 방지용
from sklearn import model_selection
cross_vali = model_selection.cross_val_score(model, data_train, label_train, cv= 5)
print('각각(5겹의) 의 검증 정확도 : ', cross_vali)
print('평균(5겹의) 의 검증 정확도 : ', cross_vali.mean())


# 분류 정확도 확인 

ac_score = metrics.accuracy_score(label_test, pred)
print('정확도: ' , ac_score) 
cl_report = metrics.classification_report(label_test,pred)
print('분류 보고서 : ' , cl_report)

# 시각화 
tbl = pd.read_csv('bmi.csv', index_col = 2)
print(tbl.head(3))

fig = plt.figure() # 이미지 저장 선언 

def scatter_func(lbl, color):
    b = tbl.loc[lbl]
    plt.scatter(b['weight'], b['height'], c= color , label = lbl)
    
scatter_func('fat','red')
scatter_func('normal','yellow')
scatter_func('thin','blue')
plt.legend()
plt.savefig('bmi_test.png')
plt.show()




    
    
    
    
    



























    