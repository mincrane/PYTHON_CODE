from share_sys_module import *

def graph_model_v1(datin ,spliter , den  ,num ,bin=0,gtype= '',title='' ):
    
    '''
    Graph precision /recall curve
    
    
    graph_model_v1(boc_20190327_new,'bocv2_score','','perf_bad_30d' ,bin=0,title = 'perf_bad_30d')
    
    '''
    
    if len(spliter)==0 | len(num) == 0:
        raise ValueError("Spliter or numerator cannot be zeor or missing")
        
    if bin >0 and bin< 10:
        raise ValueError("add bin number GE 10 to smooth the line ")
    
    #copy input data
    if len(den) == 0:
        df = datin[[spliter,num]].copy()
    else:
        df = datin[[spliter,den,num]].copy()
    
    
    #group by bins
    if bin>=10:        
        fac=bin_cut(df[spliter],bin) 
        df[spliter] = fac

    
    # group by all scores
    if len(den) == 0:
        df['Total'] = np.ones(len(df[spliter]))
        bad_rate = np.mean(datin[num])
       
        s=df.groupby([spliter])['Total',num].sum().fillna(0)
        s.rename(columns={num:'Sum_Bad'},inplace=True)
    else:
        s=df.groupby([spliter])[den,num].sum().fillna(0)
        s.rename(columns={den:'Total',num:'Sum_Bad'},inplace=True)
        bad_rate = datin[num].sum()/datin[den].sum()

    print(bad_rate)    
    s['Sum_Good'] = s['Total']-s['Sum_Bad']
    sum_tot_bad = np.sum(s.Sum_Bad)
    sum_tot_good = np.sum(s.Sum_Good)
    
    s['good_pct'] = s['Sum_Good']/sum_tot_good
    s['bad_pct'] = s['Sum_Bad']/sum_tot_bad
    
    s.sort_index(inplace=True,ascending =False)

    s[['cum_tot','cum_bad','cum_good','cum_pct_gd','cum_pct_bd']] = s.cumsum()

    s['precision']=s['cum_bad']/s['cum_tot']
    s['recall'] = s['cum_pct_bd']
   
    display(s.head(20))
    #display(s.tail(20))
    
    #calculate GINI
    zeros = pd.DataFrame([[0,0]],columns=['cum_pct_gd','cum_pct_bd'])
    gini = zeros.append(s[['cum_pct_gd','cum_pct_bd']],ignore_index=True)
    
    #### plot GINI and PR
    plt.figure(figsize=(10, 4))
    
    
    plt.subplot(121)

    plt.step(s.recall, s.precision, color='b', alpha=0.9, where='post', label='Total: '+ spliter)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.ylim([0.0, 1.05])
    plt.ylim([0.0, 0.2])
    plt.xlim([0.0, 1.05])
    plt.axhline(y= bad_rate, color='r', linestyle='-',label='Random')
   
    plt.title('Precision-Recall Curve: ' + title +'\n bad_rate={0:0.2f}'.format(bad_rate))
    
    plt.legend()



    plt.subplot(122)
    plt.title('GINI Curve: '+title)
    plt.plot(gini.cum_pct_gd,gini.cum_pct_bd,'b-',label='Total: ' + spliter)


    plt.plot([0,1], [0,1], 'k--',label='Random')
    plt.legend(loc='lower right')
    plt.xlabel("% Cumulative Good", fontweight='bold')
    plt.ylabel("% Cumulative Bad", fontweight='bold')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])


    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.40)
    plt.plot()
    plt.show()


    
    
    
    
    if gtype =='PR':
        print('\n')
        print("Precision/Recall:",'\n')
        plt.plot(s.recall,s.precision,'y')
    
    #plot gini
    if gtype == 'GINI':
        
        print('\n')
        print("GINI Curve:",'\n')
        
        
        
        plt.plot(gini.cum_pct_gd,gini.cum_pct_bd,'r')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        
        
def graph_model_scores(datin , score , den,num ,bin=0):
    
    '''
    compare scores using GINI and Precision/Recall
    
    graph_model_scores(datin = boc_20190327_final,score=['bocv1_score','bocv2_score'],den = '',num = 'perf_bad_30d',bin=400)

    
    '''
    
    s1 = score[0]
    s2 = score[1]
    
    sr1_pr,sr1_gini = graph_model_(datin,s1,den,num ,bin=400)
    sr2_pr,sr2_gini = graph_model_(datin,s2,den,num ,bin=400) 
    
    if len(den) == 0:
        bad_rate = np.mean(datin[num])
    else:
        bad_rate = datin[num].sum()/datin[den].sum()
    
    
    
    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.title('Precision-Recall Curve: '+num)
  
    plt.step(sr1_pr.recall, sr1_pr.precision, color='r', alpha=0.9, where='post', label=s1)
    plt.step(sr2_pr.recall, sr2_pr.precision, color='y', alpha=0.9, where='post', label=s2)

    

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 0.15])
    plt.xlim([0.0, 1.05])
    plt.axhline(y= bad_rate, color='b', linestyle='-',label='Random')

    plt.legend()



    plt.subplot(122)
    plt.title('GINI Curve: ' + num)
    plt.plot(sr1_gini.cum_pct_gd,sr1_gini.cum_pct_bd,'b-',label=s1)
    plt.plot(sr2_gini.cum_pct_gd,sr2_gini.cum_pct_bd,'r-',label=s2) 


    plt.plot([0,1], [0,1], 'k--',label='Random')
    plt.legend(loc='lower right')
    plt.xlabel("% Cumulative Good", fontweight='bold')
    plt.ylabel("% Cumulative Bad", fontweight='bold')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])


    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.40)
    plt.plot()
    plt.show()   


    
    
