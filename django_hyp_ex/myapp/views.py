from django.shortcuts import render
from myapp.models import SurveyData
import pandas as pd 
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Create your views here.



def MainFunc(request):
    
    return render(request,'main.html')


def listFunc(request):
    
    return render(request,'list.html')

def SurveyProcess(request):
    insertData(request)
    rdata = list(SurveyData.objects.all().values())
    df = pd.DataFrame(rdata)
    df2 = pd.DataFrame(rdata)
    #print(df)
    #검정 1 : 성별에 따라  게임 사용시간 평균에 차이가 있는가?  <== t-test
    
    man = df[df['gender'] == '남']['game_time']
    woman = df[df['gender'] == '여']['game_time']    
    
    p_sample = stats.ttest_ind(man,woman)
    #print(p_sample)
    pv = p_sample.pvalue
    #print(pv, '내방법')
       
    
    if pv >= 0.05:
        results = 'p값이 {0} > 0.05 이상이므로  성별에 따라 게임사용시간 평균 차이는 크게없다.(귀무)'.format(pv)
    else:
        results = 'p값이{0} < 0.05 미만이므로  성별에 따른 게임사용시간 평균 차이는 있다.(대립가설)'.format(pv)
    
    
    # 검정 2 : 직업에 따라  게임 사용시간 평균에 차이가 있는가?  <== oneway ANOVA
    
    #print(df)
    group1 = df[df['job'] == '화이트칼라']['game_time']    
    group2 = df[df['job'] == '블루칼라']['game_time']  
    group3 = df[df['job'] == '학생']['game_time']  
    group4 = df[df['job'] == '기타']['game_time']  
   
    #print(group1)
    
    
    f_sta,p_val = stats.f_oneway(group1,group2,group3,group4)
    #print('결과1 : f_sta:{}, p_val:{}'.format(f_sta,p_val))
    
    if p_val >= 0.05:
        results2 = 'p값이 {0} > 0.05 이상이므로  직업에따른 게임 사용시간 평균 차이는 크게없다.(귀무)'.format(p_val)
    else:
        results2 = 'p값이{0} < 0.05 미만이므로  직업에따른 게임 사용시간 평균 차이는 크게없다.(대립가설)'.format(p_val)
    
    plt.rc('font',family='malgun gothic')   #한글 깨짐 방지.
    plt.rcParams['axes.unicode_minus'] = False   # -부호 깨짐 방지
    
    
    plt.gcf()
    sns.barplot(data=df2 , x='gender',y='game_time')
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '\\static\\abc.jpg')
    plt.close()
    
    
    mean = df['game_time'].groupby(df['job']).mean() # 직업별 게임시간의 차이  
    print(mean)
    plt.gcf()
    plt.pie(mean,labels=['기타','블루칼라','학생','화이트칼라'], autopct='%0.1f%%')
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + '\\static\\bbb.jpg')
    plt.close()
    
    return render(request,'result.html',{'result': results,'result2': results2})
    

def insertData(request):
#   print(request.POST.get('gender'))
#   print(request.POST.get('age'))
#   print(request.POST.get('co_survey'))
    if request.method == 'POST':
        
        SurveyData(
            # rnum = len(list(Survey.objects.all().values())) + 1 # 자동증가 칼럼이 아니면 직접증가
            job = request.POST.get('job'),
            gender = request.POST.get('gender'),
            game_time = request.POST.get('playgame'),
                       
            ).save()
            

