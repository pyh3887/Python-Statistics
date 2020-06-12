import scipy.stats as stats
import pandas as pd 
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt 
import urllib.request
import MySQLdb
config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'123',
    'database':'test',
    'port':3306,
    'charset':'utf8',
    'use_unicode':True
}

# [ANOVA 예제 1]
# 
# 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
# 
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
# 
# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.

# 귀무 : 기름의 종류에 따라 흡수하는 기름의 평균의 차이는 영향은 없다.
# 대립 : 기름의 종류에 따라 흡수하는 기름의 평균의 차이는 영향은 있다.  

data = pd.read_csv('../testdata/bread.csv')

data.columns = ['kind','bread']


data['bread'] = data['bread'].fillna(data['bread'].mean())
print(data)

gr1 = data[data['kind'] == 1]
gr2 = data[data['kind'] == 2]
gr3 = data[data['kind'] == 3]
gr4 = data[data['kind'] == 4]

print(gr1)

print(stats.shapiro(gr1['bread'])[1]) #0.7160  > 0.05  정규성을띔
print(stats.shapiro(gr2['bread'])[1]) #0.59239  > 0.05  정규성을띔
print(stats.shapiro(gr3['bread'])[1]) #0.486010  > 0.05  정규성을띔
print(stats.shapiro(gr4['bread'])[1]) #0.41621  > 0.05  정규성을띔


# f_statistic, p_val = stats.f_oneway(gr1,gr2,gr3,gr4)
# print('일원분산분석 결과 : f_statistic:%f , p_val:%f'%(f_statistic,p_val))

lmodel = ols('bread ~ C(kind)', data).fit() # 0.870701 > 0.05 이므로 귀무채택  
print(anova_lm(lmodel))

#결과 : 기름의 종류에 따라 흡수하는 기름의 평균차이는 영향이 없다.



# [ANOVA 예제 2]
# 
# DB에 저장된 buser와 jikwon 테이블을 이용하여 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있는지 검정하시오.
# 
# 만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.

#귀무 가설 : 부서별 연봉평균에 차이가 없다.
#대립 가설 : 부서별 연봉평균에 차이가 있다.

conn = MySQLdb.connect(**config)

cursor = conn.cursor()

sql = '''
select buser_name,jikwon_pay from jikwon join buser on buser_num = buser_no 
'''
cursor.execute(sql)


df2 = pd.read_sql(sql,conn)
df2.columns =['buser','pay']
df2.loc[df2['buser'] == '총무부', 'buser'] = 1
df2.loc[df2['buser'] == '영업부', 'buser'] = 2
df2.loc[df2['buser'] == '전산부', 'buser'] = 3
df2.loc[df2['buser'] == '관리부', 'buser'] = 4



df2['pay'] = pd.to_numeric(df2['pay'])
df2['buser']
print(df2)
gr1 = df2[df2['buser'] == 1]
gr2 = df2[df2['buser'] == 2]
gr3 = df2[df2['buser'] == 3]
gr4 = df2[df2['buser'] == 4]




print(stats.shapiro(gr1['pay'])[1]) # 0.02604 < 0.05  정규성x
print(stats.shapiro(gr2['pay'])[1]) # 0.0256 < 0.05  정규성x
print(stats.shapiro(gr3['pay'])[1]) # 0.41940   > 0.05  정규성을띔
print(stats.shapiro(gr4['pay'])[1]) # 0.90780   > 0.05  정규성을띔

lmodel = ols('pay ~ C(buser)', df2).fit() # 0.745442 > 0.05 이므로  결론 : 귀무채택 부서별 연봉평균에 차이가 없다.
print(anova_lm(lmodel))






