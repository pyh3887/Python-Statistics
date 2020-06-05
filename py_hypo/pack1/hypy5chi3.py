# 실습 ) 국가전체와 지역에 대한 인종 간 인원수로 독립성 검정 실습 
# 두집단(국가전체-national, 특정지역 - la)의 인종간 인원수의 분포가 관련이 있는가 ?

import numpy as np
import pandas as pd 
import scipy.stats as stats



national = pd.DataFrame(["white"] * 100000 + ["hispanic"] * 60000 + ["black"] * 50000 + ["asian"] * 15000 + ["other"] * 35000)
la = pd.DataFrame(["white"] * 600 + ["hispanic"] * 300 + ["black"] * 250 + ["asian"] * 75 + ["other"] * 150)
#print(national)
#print(la)

#방법 1 함수 이용

na_table = pd.crosstab(index = national[0],columns='count')
#print(na_table)

l_pable = pd.crosstab(index = national[0],columns='count')
print(na_table.count())
la_table = pd.crosstab(index = la[0] ,columns='count')
print(na_table)

na_table['count_la'] = la_table['count']
#chi2, p , df : 18.09952 4
#해석 : p(0.0011) <0.05 귀무기각

# 방법2 : pvalue 구하기
# 검정통계량 계삭식 sum((관측값 기대값 ^2 기대합}

print('---------------')
# 방법2 : pvalue 구하기
# 겁정토계량  sum((관측값 기대값 ^2 기대합}
observed = la_table # 관측값
national_ratio = na_table / len(national) #기대값
expected = national_ratio * len(la) #기대값
print('expected :' , expected)

chi_sqred_stat = (((observed - expected) ** 2)/ expected).sum()
print('chi_sqared_stat :' , chi_sqred_stat)
#p-value 
pv = 1-stats.chi2.cdf(x= chi_sqred_stat, df= 4)
print('p-value:', pv )









