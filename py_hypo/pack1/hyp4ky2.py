# 이원 카이제곱 - 교차분할표 이용 : 두 개 이상의 변인(집단 또는 범주)를 대상으로 검정을 수행한다.
# 분석대상의 집산 수에 의해서 독립성 검정과 동질성 검정으로 나뉜다.
# 독립성(관련성) 검정

# - 동일 집단의 두변인 (학력수준과 대학진학 여부)를 댇상으로 관련성이 있는가 없는가 ?
# - 독립성 검정은 두 변수 사이의 연관성을 검정한다 .

# 실습 : 교육수준과 흡연율 간의 관련성 분석 : smoke.csv
# 귀무 :교육수준과 흡연율 간의 관련이 없다. (독립이다)
# 대립 :교육수준과 흡연율 간의 관련이 있다.

import pandas as pd 
import scipy.stats as stats 

data = pd.read_csv('../testdata/smoke.csv')
print(data)

print(data['education'].unique()) #[1 2 3] 대학원졸 대졸 고졸
print(data['smoking'].unique()) #[1 2 3] 과흡연 흡연 비흡연 

# 교차 분할표 

ctab = pd.crosstab(index=data['education'], columns = data['smoking']) #크로스테이블
ctab.index = ['대학원졸','대졸','고졸']
ctab.columns = ['과흡연','흡연','비흡연']
print(ctab)

chi_result =[ctab.loc['대학원졸'],ctab.loc['대졸'],ctab.loc['고졸']]

chi2, p, df , _ = stats.chi2_contingency(chi_result)
#chi2, p, df , _ = stats.chi2_contingency(ctab)
msg = '결과 chi2:{}, p:{}, df:{}'
print(msg.format(chi2, p, df , _))

# 결과 chi2:18.910915739853955, p:0.0008182572832162924, df:4

# 해석 : p:0.00081 < 0.05 이므로 귀무가설 기각 . 교육수준과 흡연율간의 관련이 있다.

# 야트보정이 자동 처리 관련--------------------------- 
# 분할표의 자유도가 1인 경우는 x^2 값이 약간 높게 계산된다.
# 그래서 아래의 식과 같이 절대값 [o-e]










