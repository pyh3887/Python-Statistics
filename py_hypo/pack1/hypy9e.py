
import numpy as np
import scipy as sp
import scipy.stats as stats
import random
import MySQLdb
import pandas as pd

config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'123',
    'database':'test',
    'port':3306,
    'charset':'utf8',
    'use_unicode':True
}


# [two-sample t 검정 : 문제1] 
# 
# 다음 데이터는 동일한 상품의 포장지 색상에 따른 매출액에 대한 자료이다. 
# 
# 포장지 색상에 따른 제품의 매출액에 차이가 존재하는지 검정하시오.

# 귀무가설 : 매출액에 차이가없다
# 대립가설 : 매출액에 차이가 있다.

bluec = [70, 68 ,82, 78, 72, 68, 67, 68, 88, 60,80]
redc = [60, 65, 55, 58, 67, 59, 61, 68, 77, 66 ,66]
p_sample = stats.ttest_rel(bluec,redc)

df = pd.DataFrame()
df['파랑'] = bluec
df['빨강'] = redc
print(df)

print(stats.shapiro(df['파랑']),'\n')  # p-value : 0.014267252758145332   p-value가 0.05보다 크므로 정규성 만족
print(stats.shapiro(df['빨강']),'\n')  # p-value : 0.029116615653038025   p-value가 0.05보다 크므로 정규성 만족
print(stats.ttest_ind(df['파랑'],df['빨강'],equal_var=True),'\n') # 0.008316 < 0.05 이므로 동분산성 o 
print(p_sample,'\n')
print('-------'*6)
# pvalue=0.008676456444140185 < 0.05 이므로 귀무기각이므로 매출액에 차이가 있다.


# [two-sample t 검정 : 문제2]  
# 
# 아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 추출하여 혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.

#귀무 : 남녀에 따른 콜레스트롤 양의 차이는 없다
#대립 : 남녀에 따른 콜레스트롤 양의 차이는 있다

man=[ 0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3]
woman=[1.4 ,2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4]

list = random.sample(man,k=15)
list2 = random.sample(woman,k=15)

p_sample = stats.ttest_rel(list,list2)
print(p_sample)
print('-------'*6)
#pvalue=0.23643177689281514 < 0.05 이므로 귀무기각이다.  
# 결과 : 남녀에 따른 콜레스트롤 양의 차이는 있다.


# [two-sample t 검정 : 문제3]
# 
# DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오.
# 
# 연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.

# 귀무가설 :총무부 영업부직원의 연봉 평균 차이가 존재하지 않는다
# 대립가설 :총무부 영업부직원의 연봉 평균 차이가 존재한다.

conn = MySQLdb.connect(**config)

cursor = conn.cursor()

sql = '''
select buser_name,jikwon_pay from jikwon join buser on buser_num = buser_no where buser_name = '총무부' 
'''
cursor.execute(sql)

sql2 = '''
select buser_name,jikwon_pay from jikwon join buser on buser_num = buser_no where buser_name = '영업부'
'''
cursor.execute(sql2)

df2 = pd.read_sql(sql,conn)
df3 = pd.read_sql(sql2,conn)
print(df2)
print(df3)

joindata = pd.merge(df2,df3,how='outer', left_index=True, right_index=True)
joindata.iloc[:,1] = joindata.iloc[:,1].fillna(joindata.iloc[:,1].mean())

print(joindata)

print(stats.shapiro(joindata.iloc[:,1]),'\n')  # p-value : 0.014267252758145332   p-value가 0.05보다 작으므로 정규성 만족 x
print(stats.shapiro(joindata.iloc[:,3]),'\n')  # p-value : 0.029116615653038025   p-value가 0.05보다 작으므로 정규성 만족 x
print(stats.ttest_ind(joindata.iloc[:,1],joindata.iloc[:,3],equal_var=True),'\n') # pvalue=0.549639 > 0.05 이므로 동분산성 x 

p_sample = stats.ttest_rel(joindata.iloc[:,3],joindata.iloc[:,1])
print(p_sample,'\n')
print('------' * 6)

#  [대응표본 t 검정 : 문제4]
# 
# 어느 학급의 교사는 매년 학기 내 치뤄지는 시험성적의 결과가 실력의 차이없이 비슷하게 유지되고 있다고 말하고 있다. 
# 
# 이 때, 올해의 해당 학급의 중간고사 성적과 기말고사 성적은 다음과 같다. 점수는 학생 번호 순으로 배열되어 있다.

# 그렇다면 이 학급의 학업능력이 변화했다고 이야기 할 수 있는가?
# 귀무가설 : 학업능력은 변화하지 않았다.
# 대립가설 : 학업능력은 변화하였다.

mid =[ 80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80]
final = [90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95]


exam= pd.DataFrame(data = mid,columns = ['mid'])
exam['final'] = final
print(exam)


print(stats.shapiro(exam['mid']),'\n')  # p-value :0.368146   p-value가 0.05보다 크므로 정규성 만족 
print(stats.shapiro(exam['final']),'\n')  # p-value : 0.19300280  p-value가 0.05보다 크므로 정규성 만족 

print(stats.ttest_rel(mid,final))
# pvalue=0.02348619 < 0.05 이므로 귀무가설 기각 학업능력은 변화햐였다.













