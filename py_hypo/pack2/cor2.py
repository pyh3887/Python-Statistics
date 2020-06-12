# 미국 , 일본,  중국 사람들의 한국 관강지 선호 지역 상관관계 분석

import json 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
plt.rc('font',family='malgun gothic')

def setScatterCorr(tour_table,all_table,tourpoint):    
    #print(tour_table,all_table,tourpoint) #창덕궁 운현궁 경복궁 창경궁
    # 계산할 관광지명에 해당하는 데이터만 뽑아 tour에 저장하고, 외국인 관광객 자료와 병합    
    tour = tour_table[tour_table['resNm']== tourpoint]
    #print(tour)
    merge_table = pd.merge(tour,all_table,left_index = True,right_index=True)
    #print(merge_table)
    
    #시각화 
    plt.subplot(1,3,1)
    plt.xlabel('중국인 입장수')
    plt.ylabel('외국인 입장객수')
    
    # 상관계수 r 얻기  중국 
    lamb1 = lambda p:merge_table['china'].corr(merge_table['ForNum'])
    r1 = lamb1(merge_table)
    plt.title('r:{:.5f}'.format(r1)) # 상관계수 제목에 표시
    plt.scatter(merge_table['china'],merge_table['ForNum'],s= 6, c='black')
  
    
    #시각화 
    plt.subplot(1,3,1)
    plt.xlabel('일본인 입장수')
    plt.ylabel('외국인 입장객수')
    
    # 상관계수 r 얻기 일본
    lamb1 = lambda p:merge_table['japan'].corr(merge_table['ForNum'])
    r2 = lamb1(merge_table)
    plt.title('r:{:.5f}'.format(r2)) # 상관계수 제목에 표시
    plt.scatter(merge_table['china'],merge_table['ForNum'],s= 6, c='black')

    
    #시각화 
    plt.subplot(1,3,1)
    plt.xlabel('미국인 입장수')
    plt.ylabel('외국인 입장객수')
    
    # 상관계수 r 얻기 
    lamb1 = lambda p:merge_table['usa'].corr(merge_table['ForNum'])
    r3 = lamb1(merge_table)
    plt.title('r:{:.5f}'.format(r3)) # 상관계수 제목에 표시
    plt.scatter(merge_table['china'],merge_table['ForNum'],s= 6, c='black')
    

    
    return [tourpoint ,r1,r2,r3]
    

def Gogo():
    fname = '서울특별시_관광지입장정보_2011_2016.json'
    jsonTP = json.loads(open(fname,'r',encoding='utf-8').read())
    #print(frame,type(frame)) #<class 'list'>
    tour_table = pd.DataFrame(jsonTP,columns=('yyyymm','resNm','ForNum')) # '년월일' , '관광지명' , '
    tour_table = tour_table.set_index('yyyymm')
    #print(tour_table) # 201101 창덕궁 14137
    
    #관광지 이름 얻기 
    resNm = tour_table.resNm.unique() # 관광지 이름 얻기 
    #print('관광지 이름:' ,resNm) #['창덕궁' '운현궁' '경복궁' ... '롯데월드']
    print('대상 관광지 이름 :' ,resNm[:5])  #대상 관광지 이름 : ['창덕궁' '운현궁' '경복궁' '창경궁' '종묘']
    
    # 중국인 관광객 정보 
    cdf = '중국인방문객.json'
    jdata = json.loads(open(cdf,'r',encoding='utf-8').read())
    china_table = pd.DataFrame(jdata,columns = ('yyyymm','visit_cnt'))
    china_table = china_table.rename(columns = {'visit_cnt':'china'})
    china_table = china_table.set_index('yyyymm')
    print(china_table.head(3))
    
        #일본인 관광객 정보 
    jdf = '일본인방문객.json'
    jdata = json.loads(open(jdf,'r',encoding='utf-8').read())
    japan_table = pd.DataFrame(jdata,columns = ('yyyymm','visit_cnt'))
    japan_table = japan_table.rename(columns = {'visit_cnt':'japan'})
    japan_table = japan_table.set_index('yyyymm')
    print(japan_table.head(3))
    
    udf = '미국인방문객.json'
    jdata = json.loads(open(udf,'r',encoding='utf-8').read())
    usa_table = pd.DataFrame(jdata,columns = ('yyyymm','visit_cnt'))
    usa_table = usa_table.rename(columns = {'visit_cnt':'usa'})
    usa_table = usa_table.set_index('yyyymm')
    print(usa_table.head(3))
    
    
    # merge 
    all_table =pd.merge(china_table, japan_table ,left_index=True , right_index = True)
    all_table =pd.merge(all_table, usa_table ,left_index=True , right_index = True)
    print(all_table.head(3))
    
    r_list = [] # 각 관광지(5군데) 마다 상관계수를 구해 기억
    for tourpoint in resNm[:5]:
        #print(tourpoint)            
        # 시각화 + 상관계수 처리함수를 호출       
        
        r_list.append(setScatterCorr(tour_table, all_table, tourpoint))
        
    print(r_list)
    
    r_df = pd.DataFrame(r_list, columns = ('고궁명','중국','일본','미국'))
    r_df = r_df.set_index('고궁명')
    print(r_df)
    
    r_df.plot(kind = 'bar' , rot =50)
    plt.show()
if __name__=='__main__':
    Gogo()
