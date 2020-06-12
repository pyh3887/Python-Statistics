import seaborn as sns
import pandas as pd
import scipy.stats as stats



tips = sns.load_dataset("tips") #seaborn 으로 데이터 가져옴

print(tips)

ctab = pd.crosstab(index=tips['smoker'],columns= tips['sex']) # crosstable 을 사용해서 남녀 흡연 여부를 구함
print(ctab)
x = stats.binom_test([54,33], p=0.6,alternative='greater') # 방향성이 있기 때문에 greater
print(x) 

# pvalue 0.39070 > 0.05 귀무 채택. 60% 이상이라고 할수 없다

tips = tips[tips['time']=='Dinner']
print(tips)
 
ctab = pd.crosstab(index=tips['smoker'],columns= tips['sex'])
print(ctab)
x = stats.binom_test([29,23], p=0.8,alternative='greater')
print(x) 

# 0.99998 < 0.05 귀무 채택 비흡연자가 흡연자 보다 많다고 할수 없다