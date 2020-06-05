import pandas as pd 
import scipy.stats as stats
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



#문제 1.  부모학력 수준이 자녀의 진학여부와 관련이 있는가?를 가설검정하시오

# 귀무가설 : 부모의 학력수준은 자녀의 진학여부와 관련이 없다
# 대립가설 : 부모의 학력수준은 자녀의 진학여부와 관련이 있다


data = pd.read_csv('../testdata/cleanDescriptive.csv')
print(data)

data2 = data.dropna(axis=0)
print(data2)



ctab = pd.crosstab(index=data2['level'], columns = data2['pass']) #크로스테이블
print(ctab)

chi2, p, df , _ = stats.chi2_contingency(ctab)

msg = '결과 chi2:{}, p:{}, df:{}'
print(msg.format(chi2, p, df , _),'\n') 

# 결과 chi2:7.794459691049416, p:0.02029806240489237, df:2 

# 0.02029 < 0.05 이므로 귀무가설기각이므로 부모의 학력수준은 자녀의 진학여부와 관련이 다.




#문제 2 . 직원의 직급과 직원의 연봉에 대한 관련성

#귀무가설 : 직급과 연봉은 관련이 없다. 
#대립가설 : 직급과 연봉은 관련이 있다.


conn = MySQLdb.connect(**config)

cursor = conn.cursor()

sql = '''
SELECT jikwon_jik,
   (CASE WHEN jikwon_pay>= '7000' THEN 4
        WHEN (jikwon_pay>= '5000' AND jikwon_pay < '7000') THEN 3
        WHEN (jikwon_pay>= '3000' AND jikwon_pay < '5000') THEN 2 
        WHEN (jikwon_pay>= '2000' AND jikwon_pay < '3000') THEN 1
        
    END) AS jikwon_pay
FROM jikwon
'''

df2 = pd.read_sql(sql,conn)
print(df2[['jikwon_jik','jikwon_pay']],'\n')


ctab = pd.crosstab(index=df2['jikwon_jik'], columns = df2['jikwon_pay']) #크로스테이블


chi2, p, df , _ = stats.chi2_contingency(ctab)
msg = '결과 chi2:{}, p:{}, df:{}'

print(msg.format(chi2, p, df , _),'\n') 

#결과 chi2:120.0, p:0.04917667372448821, df:96 

# 0.00019211533885 < 0.05 이므로 귀무기각이다. 그러므로 직급과 연봉은 관련이 있다.









