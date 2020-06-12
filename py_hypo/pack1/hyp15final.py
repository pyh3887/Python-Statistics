#db 자료를 읽어 이를 근거로 가설검정 정리 

import MySQLdb
import pandas as pd 
import numpy as np 
import ast 
import scipy.stats as stats

with open('mariadb.txt', 'r') as f:
    config = f.read()
    
config = ast.literal_eval(config)
#print(config)

conn = MySQLdb.connect(**config)
cursor = conn.cursor()

sql = '''
    select jikwon_no, jikwon_name,jikwon_jik,jikwon_pay
    from jikwon
    where jikwon_jik = '과장'
    '''

cursor.execute(sql)

for data in cursor.fetchall():
    print('%s %s %s %s'%data)
    
print('\n------ 교차분석(이원 카이제곱) ----------') #요인이두개 교차표 사용함 
df = pd.read_sql('select * from jikwon', conn)
print(df.head(), df.shape) #(30,8)

print('각 부서(범주형)와 직업 평가점수(범주형) 간의 관련성 분석- 귀무: 관련이 없다.')
buser = df['buser_num']
rating = df['jikwon_rating']

ctab = pd.crosstab(buser,rating)# 교차표 작성
print(ctab)
chi, p , df, exp = stats.chi2_contingency(ctab)
print('chi:{:.3f} , p:{:.3f} , df:{}'.format(chi,p,df,exp))
# chi:7.339 , p:0.291 , df:6
# 해석1: 카이제곱표에서 임계치 12.59 >7.339 이므로 귀무 채택
# 해석2: p= 0.291 > 0.05 이므로 귀무 채택 

print('\n-- 두 집단 이하의 평균 차이분석(t 검정) 독립:범주 , 종속:연속 -----')
print('10,20번 부서 간 평균 연봉 차이 여부를 검정 : 귀무 > 두 부서간 연봉 평균에 차이가 없다')

df_10 = pd.read_sql('select buser_num,jikwon_pay from jikwon where buser_num = 10', conn)
df_20 = pd.read_sql('select buser_num,jikwon_pay from jikwon where buser_num = 20', conn)

print(df_10.head(2))
print(df_20.head(2))
buser10 = df_10['jikwon_pay']
buser20 = df_20['jikwon_pay']

t_result = stats.ttest_ind(buser10,buser20)
print(t_result)
# Ttest_indResult(statistic=0.4585177708256519, pvalue=0.6523879191675446)
print(np.mean(buser10), ' ', np.mean(buser20)) #5414.285714285715   4908.333333333333
# 해석 : pvalue = 0.6455 > 0.05 이므로 귀무 채택
# 두부서 평균은 통계적으로 차이가 없다.

print('\n-- 세 집단 이하의 평균에 대한 분산분석(Anova,f검정) 독립:범주 , 종속:연속 -----')
print('각 부서간(4개) 평균 연봉 차이 여부를 검정: 귀무: 각 부서 간 연봉 평균에 차이가 없다 ')
df3 = pd.read_sql('select * from jikwon',conn)
print(df3.head(2))
group1 = df3[df3['buser_num'] == 10]['jikwon_pay']
print(group1[:2])
group2 = df3[df3['buser_num'] == 20]['jikwon_pay']
group3 = df3[df3['buser_num'] == 30]['jikwon_pay']
group4 = df3[df3['buser_num'] == 40]['jikwon_pay']

from statsmodels.formula.api import ols
import statsmodels.api as sm 
import matplotlib.pyplot as plt

#데이터 분포를 시각화 
#plot_data = [group1,group2,group3,group4]

#plt.boxplot(plot_data)
#plt.show()

# 일원분산분석 1 
f_sta,p_val = stats.f_oneway(group1,group2,group3,group4)
print('결과1 : f_sta:{}, p_val:{}'.format(f_sta,p_val))
# f_sta: 0.4191067001 , p_val:0.740796
# 해석 : p_val: 0.740796897 > 0.05 이므로 귀무채택

print()

# 일원분산분석 2 
lmodel = ols('jikwon_pay ~ C(buser_num)', data=df3).fit()
table = sm.stats.anova_lm(lmodel, type=1)
print(table)
# 해석 : PR(>F)(p_val):0.745442 > 0.05 이므로 귀무채택

# 사후검정 
from statsmodels.stats.multicomp import pairwise_tukeyhsd
result = pairwise_tukeyhsd(df3.jikwon_pay, df3.buser_num)
print(result)
result.plot_simultaneous()
plt.show()

#참고 : 시각화 저장 
#plt.savefig(os.path.dirname(os.path.realpath(__file__))+ '\\static\')







 



































