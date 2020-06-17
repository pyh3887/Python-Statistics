import statsmodels.api 
import statsmodels.formula.api as smf
import numpy as np 
from scipy import stats
import pandas as pd 
import matplotlib.pyplot as plt
# 회귀분석 문제 2) 
# 
# github.com/pykwon/python에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
# 
# 이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.
# 
#   - 국어 점수를 입력하면 수학 점수 예측
# 
#   - 국어, 영어 점수를 입력하면 수학 점수 예측



plt.rc('font', family = 'malgun gothic')

score = pd.read_csv('../testdata/student.csv')

#print(score)
x = score.국어 # 독립 변수
y = score.영어 # 독립 변수

result = smf.ols(formula = '수학~ 국어', data=score).fit()

print(result.summary().tables[1])
print('국어 점수를 입력하세요')
x = float(input())

print('수학 점수 예측 :' ,0.5705 * x + 32.1069) #예측 

#-----------------------------

result = smf.ols(formula = '수학~ 국어 + 영어', data=score).fit()

print(result.summary().tables[1])
print('국어 점수를 입력하세요')
x = float(input())
print('영어 점수를 입력하세요')
y = float(input())

print('수학 점수 예측 :' , (0.1158 * x) + (0.5942 * y) + 22.6238 ) #예측 




# 회귀분석 문제 3) 
# 
# 원격 DB의 jikwon 테이블에서 근무년수에 대한 연봉을 이용하여 회귀분석 모델을 작성하시오.
# 
# 장고로 작성한 웹에서 근무년수를 입력하면 예상연봉이 나올 수 있도록 프로그래밍 하시오.



