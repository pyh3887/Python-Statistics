


import pandas as pd
import scipy.stats as stats

# [one-sample t 검정 : 문제1]  
# 
# 영사기에 사용되는 구형 백열전구의 수명은 250시간이라고 알려졌다. 
# 
# 한국연구소에서 수명이 50시간 더 긴 새로운 백열전구를 개발하였다고 발표하였다. 
# 
# 연구소의 발표결과가 맞는지 새로 개발된 백열전구를 임의로 수집하여 수명시간을 수집하여 다음의 자료를 얻었다. 
# 
# 한국연구소의 발표가 맞는지 새로운 백열전구의 수명을 분석하라.

# 귀무가설 : 300시간으로 늘어나지 않았다.
# 대립가설 : 300시간으로 늘어났다!

obj1= [305,280,296,313,287,240,259,266,318,280,325,295,315,278] 
data = pd.DataFrame(obj1,columns=['수명시간'])
print(data.head())
print(data.describe())

result2 = stats.ttest_1samp(data.수명시간, popmean = 300) # 국어의 평균점수는 ? 
print('result2 :' , result2,'\n\n\n')

#result2 : Ttest_1sampResult(statistic=-1.556435658177089, pvalue=0.143606254517609)

# pvalue 0.14> 0.05 이므로 귀무 채택이다.

#--------------------------------------

#[one-sample t 검정 : 문제2] 

#국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간으로 파악되었다. A회사에서 생산된 노트북 평균시간과 차이가 있는지를 검정하기 위해서 A회사 노트북 150대를 랜덤하게 선정하여 검정을 실시한다.

comdata = pd.read_csv('../testdata/one_sample.csv')
#print(comdata.head())
#print(comdata.describe())

comdata = comdata.iloc[:,3].str.strip()
comdata = pd.to_numeric(comdata)
comdata = comdata.dropna()
print(comdata)

result2 = stats.ttest_1samp(comdata, popmean = 5.2) # 국어의 평균점수는 ? 
print('result2 :' , result2)
#result2 : Ttest_1sampResult(statistic=3.9460595666462432, pvalue=0.00014166691390197087)
# 결과값 pvalue 0.00014 < 0.05 이므로 귀무기각이다 
# 그러므로 생산된 노트북 평균시간과 차이가 있다.



#--------------------------------------

#[one-sample t 검정 : 문제2] 
# [one-sample t 검정 : 문제3] 
# 
# 
# http://www.price.go.kr에서 메뉴 중  가격동향 -> 개인서비스요금 -> 조회유형:지역별, 품목:미용 자료를 파일로 받아 미용요금을 얻도록 하자. 
# 
# 정부에서는 전국평균 미용요금이 15000원이라고 발표하였다. 이 발표가 맞는지 검정하시오.

hairdata = pd.read_excel('../testdata/hair.xls')
print(hairdata.head())
print(hairdata.describe())
print(hairdata)

pd.to_numeric(hairdata)


result2 = stats.ttest_1samp(hairdata.서울, popmean = 15000) # 국어의 평균점수는 ? 
print('result2 :' , result2)

#result2 = stats.ttest_1samp(comdata, popmean = 5.2) # 국어의 평균점수는 ? 
#print('result2 :' , result2)



















