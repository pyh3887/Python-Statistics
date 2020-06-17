
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd 
from sklearn.model_selection._split import train_test_split
# 문제 1번


# 
# data = data.drop(['요일'],axis= 1)
# 
# #print(data)
# 
# # 30 % 의 난수 로 자름
# pay = input('소득을 입력하세요\n') 
# 
# train , test = train_test_split(data, test_size = 0.3, random_state = 42)
# 
# formula = '외식유무 ~ 소득수준'
# result = smf.logit(formula = formula, data = data).fit()
# print(result)
# print(result.summary())
# 
# pred = result.predict(test)
# 
# conf_mat = result.pred_table()
# print(conf_mat)
# print('분류정확도:',(conf_mat[0][0]+ conf_mat[0][0]) / len(train))
# 
#                       
# print('예측값:' , np.rint(result.predict(test)[:5])) # 모델을 예측할때는 test
# print('실제값:' , test['외식유무'][:5])




'''=======================================================
[분류분석 문제1]

문1] 소득 수준에 따른 외식성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
다음 데이터에 대하여 소득수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
키보드로 소득수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.
========================================================'''
import pandas as pd
from sklearn.model_selection._split import train_test_split
import statsmodels.formula.api as smf
import numpy as np
from sklearn.preprocessing._data import StandardScaler
from sklearn.linear_model._logistic import LogisticRegression

crawl_data = '''토,0,57
토,0,39
토,0,28
화,1,60
토,0,31
월,1,42
토,1,54
토,1,65
토,0,45
토,0,37
토,1,98
토,1,60
토,0,41
토,1,52
일,1,75
월,1,45
화,0,46
수,0,39
목,1,70
금,1,44
토,1,74
토,1,65
토,0,46
토,0,39
일,1,60
토,1,44
일,0,30
토,0,34'''

# 데이터 정제
data = pd.read_csv('aa.csv')

# print(data)

# 주말 골라내기
data = data[data['요일'].isin(['토','일'])]
# print(data)

# 요일 컬럼 삭제
data = data.drop(['요일'], axis=1)
# print(data)

# 데이터형 변경
data = data.astype(float)

# train / test data로 분리
train, test = train_test_split(data, test_size=0.3, random_state=0)

# print(data)
# print(train)

# formula
my_formula = '외식유무 ~ 소득수준'

# 학습모델 생성
model = smf.logit(formula=my_formula, data=train).fit()

user_input = float(input('소득수준을 입력해보세요 :'))
newdf = data.iloc[:1].copy()
newdf['소득수준'] = user_input

print(newdf)

print('외식여부 \n', np.rint(model.predict(newdf)))

# print(model)
# print(model.summary())

'''=========================================================
게임, TV 시청 데이터로 안경 착용 유무를 분류하시오.
안경 : 값1(착용X), 값2(착용O)
예제 파일 : https://github.com/pykwon  ==>  bodycheck.csv
새로운 데이터(키보드로 입력)로 분류 확인
========================================================='''

# 데이터 생성
data2 = pd.read_csv('../testdata/bodycheck.csv')
# print(data2.head(5))
use_data = data2.drop(['신장', '체중'], axis=1)
# print(use_data.head(5))

x = use_data.loc[:, ['게임', 'TV시청']]   
y = use_data.안경유무

# print(x)
# train / test data로 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print()
# print("train / test :", x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 스케일링 (데이크 크기 표준화)
# StandardScaler() 평균이 0, 표준편차가 1이되도록 변환해준다.
sc = StandardScaler()
# print(sc)
sc.fit(x_train)
sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print('\n스케일링 처리 후')
# print(x_train[:3])

print('---- 분류 모델 사용---------------')
# C= 모델에 패널티(L2 정규화)를 부여함(overfitting 관련) 모델 정확도를 조정
ml = LogisticRegression(C=0.1, random_state=0)

# train data로 모델 학습
result = ml.fit(x_train, y_train)

# 모델 학습 후 객체를 저장
import pickle
fileName = 'final_model_ex.sav'
pickle.dump(ml, open(fileName, 'wb'))
ml = pickle.load(open(fileName, 'rb'))

# 입력받기
game_time = int(input('게임시간을 입력해보세요 :'))
tv_time = int(input('tv시청시간을 입력해보세요 :'))

# 새로운 값으로 예측
new_data = np.array([[game_time, tv_time]])
new_pred = ml.predict(new_data)
print('예측 결과 :', new_pred)



