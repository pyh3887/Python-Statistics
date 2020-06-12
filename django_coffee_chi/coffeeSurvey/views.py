from django.shortcuts import render
from coffeeSurvey.models import Survey
import matplotlib.pyplot as plt
plt.rc('font',family='malgun gothic')

# Create your views here.

def mainFunc(request):
    
    return render(request,'main.html')


def SurveyView(request):
    
    return render(request,'list.html')

def SurveyProcess(request):
    insertData(request) # 신규 자료 저장 
    rdata = list(Survey.objects.all().values())
    #print(Survey.objects.all().values())
    df,crossTab,results = ChiFunc(rdata)
    
    #시각화 : 커피사별 선호건수
    fig = plt.gcf()
    co_group = df['co_survey'].groupby(df['coNum']).count()    
  
    co_group.plot.bar(subplots = True, color = ['red','blue'], width=0.5)
    
    plt.xlabel('커피사')
    plt.ylabel('선호건수')
    plt.title('커피사별  선호건수')
    

    
    fig.savefig('django_coffee_chi/coffeeSurvey/static/images/vbar.png')
    
    
    return render(request,'result.html',{'crossTab':crossTab.to_html,'df':df.to_html(index=False),'results':results})

def insertData(request):
#   print(request.POST.get('gender'))
#   print(request.POST.get('age'))
#   print(request.POST.get('co_survey'))
    if request.method == 'POST':
        Survey(
            # rnum = len(list(Survey.objects.all().values())) + 1 # 자동증가 칼럼이 아니면 직접증가
            gender = request.POST.get('gender'),
            age = request.POST.get('age'),
            co_survey = request.POST.get('co_survey'),
            ).save()
            
import pandas as pd 
import scipy.stats as stats

def ChiFunc(rdata):
    #print(rdata,' ',type(rdata))
    df = pd.DataFrame(rdata)
    df.dropna()
    df['genNum'] = df['gender'].apply(lambda g:1 if g == '남' else 2)
    df['coNum'] = df['co_survey'].apply(lambda c:1 if c == '스타벅스' else 2 if c == '커피빈' else 3 if c =='이디야' else 4)
    print(df,'\n')
    
    crossTab = pd.crosstab(index=df['genNum'], columns = df['coNum'])
    print(crossTab)
    
    st,pv,_,_ = stats.chi2_contingency(crossTab)
    if pv >= 0.05:
        results = 'p값이 {0} : 0.05 이상이므로 <br> 성별에 따른 선호 커피 브랜드에는 차이가 없다(귀무)'.format(pv)
    else:
        results = 'p값이{0} : 0.05 미만이므로 <br> 성별에 따른 선호 커피 브랜드에는 차이가 있다(대립가설)'.format(pv)
        
    return df, crossTab, results
        
    
    
    
    
    
