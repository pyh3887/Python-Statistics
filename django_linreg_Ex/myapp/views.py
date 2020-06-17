from django.shortcuts import render
from myapp import models
import pandas as pd
import MySQLdb
import statsmodels.formula.api as smf

# 원격 DB의 jikwon 테이블에서 근무년수에 대한 연봉을 이용하여 회귀분석 모델을 작성하시오.
# 장고로 작성한 웹에서 근무년수를 입력하면 예상연봉이 나올 수 있도록 프로그래밍 하시오.

config = {
    'host':'192.168.0.24',
    'user':'root',
    'password':'123',
    'database':'test',
    'port':3306,
    'charset':'utf8',
    'use_unicode':True
}

# Create your views here.
def Main(request):
    return render(request, 'main.html')

def Check(request):
    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()
        sql = """
            select (substr(year(now()), 1, 4) - substr(jikwon_ibsail, 1, 4)) as '연수', jikwon_pay from jikwon
        """
        cursor.execute(sql)
        datas = cursor
        
    except Exception as e:
        print('err : ' + e)
    finally:
        cursor.close()
        conn.close()

    df = pd.DataFrame(datas, columns=['연수', '연봉'])
    print(df)
    
    result = smf.ols(formula='연봉 ~ 연수', data=df).fit()
    print(result.summary())
    
    intercept = 1391.7195
    coef = 592.1637
    x = float(request.POST.get('xr'))
    print(x)
    ye = (coef * x) + intercept
    print('예측 근무 년수 :', ye)
    
    return render(request, 'check.html', {'pay':ye})