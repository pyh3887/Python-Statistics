import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 어느 한 집단의 평균은 0인지 검정하기 (난수 사용)
# 귀무 : 자료들의 평균은 0 이다
# 대립 : 자료들으 평균은 0이 아니다. 

np.random.seed(123)
mu = 0 
n = 10
x = stats.norm(mu).rvs(n)
print(x,np.mean(x))
#sns.distplot(x, kde=False, rug=True, fit = stats.norm) # 시각화 
#plt.show()

result = stats.ttest_1samp(x,popmean=0)
print('result:', result)
# result: Ttest_1sampResult(statistic=-0.6540040368674593, pvalue=0.5294637946339893)
result2 = stats.ttest_1samp(x,popmean=0.8)
print('result :', result2)   # pvalue : 0.289


# 단일 모집단의 평균에 대한 가설검정(one samples t - test)
# 실습 예제 1 )
# a 중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 처리 (국어 점수 평균검정) student.csv
data = pd.read_csv('../testdata/student.csv')
print(data.head())
print(data.describe())

result2 = stats.ttest_1samp(data.국어, popmean = 80) # 국어의 평균점수는 ? 
print('result2 :' , result2)
# result2 : Ttest_1sampResult(statistic=-1.3321801667713213, pvalue=0.19856051824785262)
# 해석 p_value 0.1985 > 0.05 # 귀무채택 . 국어점수의 평균은 80이다.

print('------'*6)
# 실습 예제 2)
# 여아 신생아 몸무게의 평균 검정 수행 babyboom.csv
# 여아 신생아의 몸무게는 평균이 2800(g)으로 알려져 왔으나 이보다 더 크다는 주장이 나왔다.
# 표본으로 여아 18명을 뽑아 체중을 측정하였다고 할 때 새로운 주장이 맞는지 검정해 보자.

data = pd.read_csv('../testdata/babyboom.csv')
print(data.head()) # 1 여아 2 남아
print(data.describe())
fdata = data[data.gender == 1]
print(fdata.head(),len(fdata))
print(fdata.describe())
print(np.mean(fdata.weight)) # 3132.4444444444443

# 정규성 확인을 위한 시각화

sns.distplot(fdata.iloc[:,2], fit = stats.norm)
plt.show()
stats.probplot(fdata.iloc[:,2], plot = plt) # Q-Q plot
plt.show()

print(stats.shapiro(fdata.iloc[:,2])) # 정규성 확인 p (0.01798) < 0.05 이므로 정규성을 따르지 않는다.
#참고 : 정규성을 띄지 않으나 집단이 하나이므로 wilcox 검정은 할수 없다.

result3 = stats.ttest_1samp(fdata.weight, popmean= 2800)
print('result3 :' ,result3)
# result3 : Ttest_1sampResult(statistic=2.233187669387536, pvalue=0.03926844173060218)
# 해석 : pvalue(0.0392) < 0.05 이므로 귀무 기각

# 여아 신생아의 몸무게는 평균이 2800g 보다 크다 . 라는 주장을 받아들임















