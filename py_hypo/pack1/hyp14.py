# 이항 검정 : 두 가지 값을 가지는 확률변수의 분포를 판단 하는데 효과적
# 예 : 10명의 자격증 시험 합격자 중 남자가 6명이었다고 할 때, '남자가 여자 보다 합격률이 높다고 할 수 있는가?'

#binom_test
import pandas as pd
import scipy.stats as stats

data = pd.read_csv("../testdata/one_sample.csv")
print(data.head(3))
print(data.survey.unique())  #[1 0]
ctab = pd.crosstab(index=data['survey'],columns= 'count')
ctab.index = ['불만족','만족']
print(ctab) #자바 교육 후 교육 만족 여부 교차표 불만족 14명 만족 136명

# 양측검정 : 방향성이 없다. 기존 80% 만족을 기준 검증 실시
# 형식 : stats.binom_test(성공또는 실패수 , 시도횟수 , 가설확률 , 검정방식)
x = stats.binom_test([136,14], p=0.8,alternative='two-sided')
print(x)  # p = 0.0006734701362867019 < 0.05 귀무 기각. 기존(80%) 만족율과 차이가 있다.
# 양측검정에서는 기존 만족율 보다 크다, 작다라는 방향성을 제시하지 않는다.

x = stats.binom_test([14,136], p=0.2,alternative='two-sided')
print(x)  #결과 동일

print()
#단축검정 : 방향성이 있다. 기존 80% 만족율 기준 검증 실시
x = stats.binom_test([136,14], p=0.8,alternative='greater')
print(x) # p : 0.0003179401921985477 < 0.05 귀무 기각. 기존 80% 만족율 이상의 효과가 있다. 
# 따라서 기본 20% 보다 불만율이 작아졌다 라고 할 수 있다.

x = stats.binom_test([14,136], p=0.2,alternative='less')
print(x)


