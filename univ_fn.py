from share_sys_module import *



def var_split(ind_x,perf_y,bin_num,showtb = True):
    
    if bin_num<= 0 | isnull(bin_num):
        raise ValueError("Bin_num cannot be zeor or missing")
    
    #fac = pd.qcut(ind_x,bin_num,duplicates='drop')
    #bins = algos.quantile(ind_x, quantiles)
    #fac, bin_num = _bins_to_cuts(ind_x, bins, precision=3, include_lowest=True,dtype=None, duplicates='drop')
    
    #bins = algos.quantile(ind_x, quantiles)
    
    if isinstance(ind_x,np.ndarray):
        ind_x = pd.Series(ind_x,name= 'Score')
        varname = ind_x.name
    else:
        varname = ind_x.name
    
    
    
   
    if is_numeric_dtype(ind_x):
        fac=bin_cut(ind_x,bin_num)
    
    else:
        if isinstance(ind_x.dtype,CategoricalDtype):
            ind_x = ind_x.cat.add_categories("Missing").fillna('Missing')
        else:
            ind_x = ind_x.fillna('missing')
        fac = ind_x
    
    s=pd.concat([ind_x,perf_y],axis=1,ignore_index=True)
    s['bins']=fac
    
    s0=s.groupby(['bins',perf_y]).size().unstack().fillna(0.00001)
    
    s0.index.rename('bin_split',inplace=True)
    
    
    s0.rename(columns={0.0:'N_good',1.0:'N_bad'},inplace=True)
    
    s1 = s0.rename_axis(varname,axis=1,inplace=True)
    

    
    s1['Total']=s1['N_good']+s1['N_bad']
    
    
    ####sub total 
    sub_total = s1.sum()
    
    
    ####pct
    #num_perf = perf_y.value_counts()
    #s1['good_pct'] = s1['N_good']/num_perf[0]
    #s1['bad_pct'] = s1['N_bad']/num_perf[1]
    
    
    s1[['good_pct','bad_pct','tot_pct']] = s1.div(sub_total,axis=1)
    #print(s1)
    
    ####cum
    s1[['cum_good','cum_bad','cum_tot','cum_pct_gd','cum_pct_bd','cum_pct_tot']] = s1.cumsum().round(3)
    s1['WOE']= np.log(s1['good_pct']/s1['bad_pct']).round(3)
    s1['bin_iv'] = ((s1['good_pct'] - s1['bad_pct'])*s1['WOE']).round(3)
    s1['KS'] = abs((s1.cum_pct_gd - s1.cum_pct_bd)).round(3)*100
    
    
    IV =s1.bin_iv.sum()
    KS=abs((s1.cum_pct_gd - s1.cum_pct_bd)).max().round(4)*100
    
    s1.style.bar(subset=['WOE'], align='mid', color=[ '#5fba7d','#d65f5f'])
    #s1.style.apply(highlight_max, subset=['KS'])
    
    ####add total
    
    sub_total['Bin_split'] = 'Total'
    
    s1['Bin_split']=s1.index
    
    
    ss=s1.append(sub_total,ignore_index=True)
    ss.set_index('Bin_split',inplace=True)
    ss['bad_rate'] = ss['N_bad']/ss['Total']
    ss.loc['Total',['cum_pct_bd','cum_pct_gd','WOE']]=0
    ss.loc['Total',['tot_pct','good_pct','bad_pct','bin_iv']]=ss[['tot_pct','good_pct','bad_pct','bin_iv']].sum()
    
   
    ss[['bad_rate','tot_pct','good_pct','bad_pct']]=ss[['bad_rate','tot_pct','good_pct','bad_pct']].apply(lambda x : x.map('{:,.2%}'.format),axis=1)
    s_out = ss[['N_good','N_bad','Total','bad_rate','tot_pct','good_pct','bad_pct','cum_pct_gd','cum_pct_bd','WOE','bin_iv','KS']]
    
    if showtb:
       if bin_num > 20:
          print('KS=',KS)
          print('IV=',IV)
          display(s_out.tail(20).style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))

       else:
          print('KS=',KS)
          print('IV=',IV)
          display(s_out.style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))
	    
    		
    		
    return KS,IV,s_out


def univall(X,perf,bin_num,excelname,showtb=False):
    X_in = X.copy()
    
    s1 =X_in.describe(percentiles=[.01, .05, .25, .75, .95,.99 ] )
    n = X_in.shape[0]
    
    s2=s1.T
    s2['Varname'] = s2.index
    s2['Missing %']=((n-s2['count'])/n).map('{:,.2%}'.format)
    
    ksiv=[]
    for column in X_in:    
        ks,iv,t_out = var_split(X_in[column],perf,bin_num= bin_num,showtb=showtb)
        s=[column,ks,iv]
        ksiv.append(s)
    ksiv=pd.DataFrame(ksiv,columns=['Varname','KS','IV'])
    
    univ_table = ksiv.merge(s2,left_on='Varname',right_on = 'Varname',how='left', indicator=True).sort_values(by='KS',ascending=False)
    
    univ_table=univ_table[['Varname', 'KS', 'IV', 'count','Missing %','mean', 'std', 'min', '1%', '5%', '25%', '50%', '75%', '95%', '99%', 'max','_merge']]
    
    
    
    if len(excelname)>0:
        univ_table.to_excel(excelname)
    
    return univ_table




#only calculate KS,IV
def univ_all(X,perf,bin_num):
    X_in = X.copy()
    ksiv=[]
    for column in X_in:    
        ks,iv = var_split(X_in[column],perf,bin_num= bin_num,showtb=False)
        s=[column,ks,iv]
        ksiv.append(s)
    ksiv=pd.DataFrame(ksiv,columns=['Varname','KS','IV'])
    #ksiv.to_excel('ksiv.xlsx')
    #print(ksiv)
    return ksiv