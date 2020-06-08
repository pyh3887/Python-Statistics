import pandas as pd
import scipy.stats as stats


data = pd.read_csv('../testdata/survey_method.csv')
print(data.head())
print(data['method'].unique())
print(data['survey'].unique())

ctab = pd.crosstab(index=data['method'], columns = data['survey'])
ctab.columns = ['매우만족','만족','보통','불만족','매우불만족']
ctab.index = ['방법1','방법2','방법3']
print(ctab)
chi2,p,df,_ =stats.chi2_contingency(ctab)
print('chi2:{} , p:{}, df:{}'.format(chi2,p,df))

# chi2:6.544667820529891 , p:0.5864574374550608, df:8
# 해석 : p(0.5864) > 0.05 귀무 채택 . 교육방법에 따른 교육생들의 만족도에 차이가 없다

print('----------------------------------')



#귀무 : 연령대별로 sns 서비스들에 이용 현황이 서로 동일
# 연구 : 연령대별로 sns 서비스들에 이용 현황이 서로 동일하지 않다

snsdata = pd.read_csv('../testdata/snsbyage.csv')
print(snsdata.head())
print(snsdata['age'].unique())
print(snsdata['service'].unique())

sns_ctab =pd.crosstab(index=snsdata['service'],columns=snsdata['age'])
print(sns_ctab)

chi2,p,df,_ =stats.chi2_contingency(sns_ctab)
print('chi2:{} , p:{}, df:{}'.format(chi2,p,df))

# chi2:102.75202494484223 , p:1.167906420421286e-18, df:8 귀무가설 기각


