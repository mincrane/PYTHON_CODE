
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import sys

from IPython.display import display, HTML
pd.set_option('display.width', 1000)
#pd.options.display.float_format = '{:,.2f}'.format
import pandas.core.algorithms as algos


from pandas.core.dtypes.common import (
    is_integer,
    is_scalar,
    is_categorical_dtype,
    is_datetime64_dtype,
    is_timedelta64_dtype,
    _ensure_int64)

from pandas.core.dtypes.missing import isnull

from pandas import (to_timedelta, to_datetime,
                    Categorical, Timestamp, Timedelta,
                    Series, Interval, IntervalIndex)



pd.options.display.max_rows = 500
pd.options.display.max_columns = 500
pd.options.display.max_colwidth = 200

pd.set_option('display.width', 1000)
from IPython.display import display, HTML
import pandas.core.algorithms as algos
from pandas.api.types import is_numeric_dtype
from pandas.api.types import CategoricalDtype

def bin_cut(x,bin_num):
    
    if bin_num<= 0 | isnull(bin_num):
        raise ValueError("Bin_num cannot be zeor or missing")
    
    if not is_numeric_dtype(x):
        raise ValueError('input variable must be numeric')
    
    #if not np.issubdtype(x.dtype, np.number): (does not work for category)
        #raise ValueError('input variable must be numeric')
    
    quantiles=[n/bin_num for n in range(0,bin_num+1)]
    #print(quantiles)
    ###bins for interval
    bin_all = algos.quantile(x, quantiles).round(4)
    
    #bins for grouping
    bins = bin_all[1:len(bin_all)]
   
    bin_all[0]=(bin_all[0]-0.001).round(4)
    if x.isnull().any():
        bin_all=np.append(-999999999,bin_all)
    
    uni_bin_all=algos.unique(bin_all)
    uni_bins = algos.unique(bins)
    #print(bins)
    #print(uni_bins)
    ids=_ensure_int64(uni_bins.searchsorted(x.round(4), side='left'))
    
    if ids[ids==len(uni_bins)].any() | isnull(x).any() :  
        ids=ids+1
        ids[ids == (len(uni_bins)+1)] = 0
    
   
    labels = IntervalIndex.from_breaks(uni_bin_all, closed='right')
    
    
    
    result = algos.take_nd(labels, ids)
    #print(pd.Series.count(result))
    
    return result


def highlight_bps(s):
        color = 'yellow'
        return 'background-color: %s' % color  


def score_dist1(datin,bin_num,class_var,perf_var_col,tran_var_col,bps_var_col,other_col,showtb=True):
    
    if bin_num<= 0 | isnull(bin_num):
        raise ValueError("Bin_num cannot be zeor or missing")
    
    if len(class_var)==0 | len(perf_var_col) == 0:
        raise ValueError("Class var or perf var cannot be zeor or missing")
                                                                
    df=datin.copy()
    ind_x = df[class_var]                                                             
                                                         
    if is_numeric_dtype(ind_x):
        if ind_x.nunique() <= 10:
            ind_x = ind_x.fillna('missing')
            fac = ind_x
        else:
            fac=bin_cut(ind_x,bin_num)
    
    else:
        if isinstance(ind_x.dtype,CategoricalDtype):
            ind_x = ind_x.cat.add_categories("Missing").fillna('Missing')
        else:
            ind_x = ind_x.fillna('missing')
        fac = ind_x
                                                                 
                                                                 
    df['bins']=fac
    #df['bins'] = pd.qcut(df[class_var],bin_num,duplicates='drop')
    
    t=df.groupby(['bins'])
    varname=perf_var_col+tran_var_col+bps_var_col+other_col
    meanvar=tran_var_col+bps_var_col  #for the format of means 
    
    #SUM
    t_sum1=pd.concat([t.size(),t[varname].sum()],axis=1)
    t_sum1.rename(columns={0:'Total_Num'},inplace=True)

    #total number
    sum_all=datin[varname].sum()
    sum_tot=pd.Series([datin.shape[0]],index=['Total_Num']).append(sum_all)
    sum_tot1=sum_tot.copy()
    
    t_sum1[class_var]=t_sum1.index
    sum_tot1[class_var]='Total'


    t_sum = t_sum1.append(sum_tot1, ignore_index=True)
    t_sum.set_index(class_var, inplace=True)

    ###### calculate KS and IV
    
    
    t_univ = t_sum.iloc[:,0:2]
    #t_univ.columns.values[1] = 'Bad_cnt' #where change var name in the original Dataframe
    t_univ.rename(columns={ t_univ.columns[1]: "Bad_cnt" },inplace=True)
    
    
    
    t_univ['good_cnt'] = t_univ['Total_Num'] - t_univ['Bad_cnt']
   
    
    ## in case log(0) or divided by 0; for WOE and IV calucalation 
    t_univ.Bad_cnt[t_univ['Bad_cnt']==0]=0.00001
  
    
    t_univ[['tot_pct','bad_pct','good_pct']] = t_univ.div(t_univ.loc['Total',:],axis=1)
    
    t_univ[['cum_tot','cum_bad','cum_good','cum_pct_tot','cum_pct_bd','cum_pct_gd']] = t_univ.cumsum().round(3)
    t_univ.loc['Total',['cum_tot','cum_bad','cum_good','cum_pct_tot','cum_pct_bd','cum_pct_gd']] = 0
    
    	
    t_univ['WOE']= np.log(t_univ['good_pct']/t_univ['bad_pct']).round(3)
       
    t_univ['bin_iv'] = ((t_univ['good_pct'] - t_univ['bad_pct'])*t_univ['WOE']).round(3)
    t_univ['KS'] = abs((t_univ.cum_pct_gd - t_univ.cum_pct_bd)).round(3)*100
    
    t_univ['bad_rate']=(t_univ['Bad_cnt']/t_univ['Total_Num'])
    t_univ[['bad_rate','tot_pct','good_pct','bad_pct']]=t_univ[['bad_rate','tot_pct','good_pct','bad_pct']].apply(lambda x : x.map('{:,.2%}'.format),axis=1)
    IV =t_univ.bin_iv.sum()
    t_univ.loc['Total',['bin_iv']] = IV
    
   
    KS=abs((t_univ.cum_pct_gd - t_univ.cum_pct_bd)).max().round(4)*100

    
    
    ######################
    t_mean=t[varname].mean().round(2)
    mean_all=df[varname].mean().round(2)
    t_mean = t_mean.append(mean_all, ignore_index=True)


    t_mean[meanvar]=t_mean[meanvar].apply(lambda x : x.map('{:,.0f}'.format),axis=1)
    
    
    
    #t_mean['perf_flag_x'] = t_mean['perf_flag_x'].map('{:,.2%}'.format)
    t_mean[perf_var_col] = t_mean[perf_var_col].apply(lambda x : x.map('{:,.2%}'.format),axis=1)
    
    #print(t_mean[meanvar].describe())

 
    if len(bps_var_col)>0:
        t_bps=t_sum[bps_var_col].div(t_sum.iloc[:,2], axis=0).mul(10000).apply(lambda x : x.map('{:,.0f}'.format),axis=1)
        t_bps.columns = pd.MultiIndex.from_arrays([t_bps.columns,['BPS'] * len(t_bps.columns)],names=('Var','Stats'))
        
    t_pct=t_sum[['Total_Num']+varname].div(sum_tot,axis=1).apply(lambda x : x.map('{:,.2%}'.format),axis=1)
    t_sum=t_sum.apply(lambda x : x.map('{:,.0f}'.format),axis=1)
        
       
    # #add second index
    t_sum.columns = pd.MultiIndex.from_arrays([t_sum.columns,['Sum'] * len(t_sum.columns)],names=('Var','Stats'))

    
    #t_mean.columns = pd.MultiIndex.from_arrays([t_mean.columns,['Bad Rate','Mean','Mean','Mean'] ],names=('Var','Stats'))
    t_mean.columns = pd.MultiIndex.from_arrays([t_mean.columns,['Bad Rate']+['Mean']*(len(t_mean.columns)-1)],names=('Var','Stats'))
    t_pct.columns = pd.MultiIndex.from_arrays([t_pct.columns,['Pct_of_Tot'] * len(t_pct.columns)], names=('Var','Stats'))
    
   
    
    if len(bps_var_col)==0:
        all_var=[t_sum,t_mean.set_index(t_sum.index),t_pct]
    else:
        all_var=[t_sum,t_mean.set_index(t_sum.index),t_bps,t_pct]
  
   
    
    t_all = pd.concat(all_var,axis=1,)
   
    t_all.sort_index(level=[0,1],axis=1,inplace=True,ascending=False)
    varnameall=['Total_Num']+varname
    final_table = t_all[varnameall]
    
    
    if showtb:
    		#print score distribution
        if len(tran_var_col+bps_var_col+other_col) == 0:
            if bin_num >=20:
                print('KS =',KS)
                print('IV=',IV)
                display(t_univ[['Total_Num','good_cnt','Bad_cnt','bad_rate','tot_pct','good_pct','bad_pct','cum_pct_gd','cum_pct_bd','WOE','bin_iv','KS']].tail(121).style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))
    
               
            else:
                print('KS =',KS)
                print('IV=',IV)
                display(t_univ[['Total_Num','good_cnt','Bad_cnt','bad_rate','tot_pct','good_pct','bad_pct','cum_pct_gd','cum_pct_bd','WOE','bin_iv','KS']].style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))
    

        else:
            if bin_num>=20:
                print('KS =',KS)
                print('IV=',IV)
                display(final_table.tail(121).style.applymap(highlight_bps,subset=pd.IndexSlice[:, [col for col in zip(bps_var_col,   ['BPS']*len(bps_var_col))]]))
                
            else :
                print('KS =',KS)
                print('IV=',IV)
                display(final_table.style.set_properties(**{
                           'align':"middle",
                           'border-color':'black'})\
                .applymap(highlight_bps,subset=pd.IndexSlice[:, [col for col in zip(bps_var_col,['BPS']*len(bps_var_col))]]))   
        
    #return final_table
    
def score_dist(datin,bin_num,class_var,perf_var_col,tran_var_col,bps_var_col,other_col,head=0,tail=0):
    '''
    score distributions of unit bad and perf metrics
    datin: Input dataframe
        bin_num: how many bins to split score. if class_var is not continuous , it will be ignored 
        class_var: score or any variables you would like to see the performance metrics by groups
        perf_var_col: unit bad
        tran_var_col: GMV
        bos_var_col: a list of net losses, claims etc.
        other_col: such as suspension flag 
        
        example:
        s = ScoreRank(test100k,10,'bocv2_score',['perf_bad_30d'],['perf_gmv_30d'],['perf_amt_open_inr_30d', 'perf_rvi_net_loss_30d','perf_amt_esc_clm_30d'],['flag_cur_susp','perf_res_30d','perf_susp_30d','perf_susp_hold_30d'])
     
     By: Mike Min 
     V1: 04/2018
     V2: 08/2019 This function is deprecated. Please use ScoreRank().
     
     python: 3.6.5
     pandas: 0.23.1
     
     '''
    
    
    if bin_num<= 0 | isnull(bin_num):
        raise ValueError("Bin_num cannot be zeor or missing")
    
    if len(class_var)==0 | len(perf_var_col) == 0:
        raise ValueError("Class var or perf var cannot be zeor or missing")
                                                                
    df=datin.copy()
    ind_x = df[class_var]                                                             
                                                         
    if is_numeric_dtype(ind_x):
        if ind_x.nunique() <= 10:
            ind_x = ind_x.fillna('missing')
            fac = ind_x
        else:
            fac=bin_cut(ind_x,bin_num)
    
    else:
        if isinstance(ind_x.dtype,CategoricalDtype):
            ind_x = ind_x.cat.add_categories("Missing").fillna('Missing')
        else:
            ind_x = ind_x.fillna('missing')
        fac = ind_x
                                                                 
                                                                 
    df['bins']=fac
    #df['bins'] = pd.qcut(df[class_var],bin_num,duplicates='drop')
    
    t=df.groupby(['bins'])
    varname=perf_var_col+tran_var_col+bps_var_col+other_col
    meanvar=tran_var_col+bps_var_col  #for the format of means 
    
    #SUM
    t_sum1=pd.concat([t.size(),t[varname].sum()],axis=1)
    t_sum1.rename(columns={0:'Total_Num'},inplace=True)

    #total number
    sum_all=datin[varname].sum()
    sum_tot=pd.Series([datin.shape[0]],index=['Total_Num']).append(sum_all)
    sum_tot1=sum_tot.copy()
    
    t_sum1[class_var]=t_sum1.index
    sum_tot1[class_var]='Total'


    t_sum = t_sum1.append(sum_tot1, ignore_index=True)
    t_sum.set_index(class_var, inplace=True)

    ###### calculate KS and IV
    
    
    t_univ = t_sum.iloc[:,0:2]
    #t_univ.columns.values[1] = 'Bad_cnt' #where change var name in the original Dataframe
    t_univ.rename(columns={ t_univ.columns[1]: "Bad_cnt" },inplace=True)
    
    
    
    t_univ['good_cnt'] = t_univ['Total_Num'] - t_univ['Bad_cnt']
   
    
    ## in case log(0) or divided by 0; for WOE and IV calucalation 
    t_univ.Bad_cnt[t_univ['Bad_cnt']==0]=0.00001
  
    
    t_univ[['tot_pct','bad_pct','good_pct']] = t_univ.div(t_univ.loc['Total',:],axis=1)
    
    t_univ[['cum_tot','cum_bad','cum_good','cum_pct_tot','cum_pct_bd','cum_pct_gd']] = t_univ.cumsum().round(3)
    t_univ.loc['Total',['cum_tot','cum_bad','cum_good','cum_pct_tot','cum_pct_bd','cum_pct_gd']] = 0
    
    	
    t_univ['WOE']= np.log(t_univ['good_pct']/t_univ['bad_pct']).round(3)
       
    t_univ['bin_iv'] = ((t_univ['good_pct'] - t_univ['bad_pct'])*t_univ['WOE']).round(3)
    t_univ['KS'] = abs((t_univ.cum_pct_gd - t_univ.cum_pct_bd)).round(3)*100
    
    t_univ['bad_rate']=(t_univ['Bad_cnt']/t_univ['Total_Num'])
    t_univ[['bad_rate','tot_pct','good_pct','bad_pct']]=t_univ[['bad_rate','tot_pct','good_pct','bad_pct']].apply(lambda x : x.map('{:,.2%}'.format),axis=1)
    IV =t_univ.bin_iv.sum()
    t_univ.loc['Total',['bin_iv']] = IV
    
   
    KS=abs((t_univ.cum_pct_gd - t_univ.cum_pct_bd)).max().round(4)*100

    
    
    ######################
    t_mean=t[varname].mean().round(2)
    mean_all=df[varname].mean().round(2)
    t_mean = t_mean.append(mean_all, ignore_index=True)


    t_mean[meanvar]=t_mean[meanvar].apply(lambda x : x.map('{:,.0f}'.format),axis=1)
    
    
    
    #t_mean['perf_flag_x'] = t_mean['perf_flag_x'].map('{:,.2%}'.format)
    t_mean[perf_var_col] = t_mean[perf_var_col].apply(lambda x : x.map('{:,.2%}'.format),axis=1)
    
    #print(t_mean[meanvar].describe())
		
 
    if len(bps_var_col)>0:
        t_bps=t_sum[bps_var_col].div(t_sum.iloc[:,2], axis=0).mul(10000).apply(lambda x : x.map('{:,.0f}'.format),axis=1)
        t_bps.columns = pd.MultiIndex.from_arrays([t_bps.columns,['BPS'] * len(t_bps.columns)],names=('Var','Stats'))
        
    t_pct=t_sum[['Total_Num']+varname].div(sum_tot,axis=1).apply(lambda x : x.map('{:,.2%}'.format),axis=1)
    t_sum=t_sum.apply(lambda x : x.map('{:,.0f}'.format),axis=1)
        
       
    # #add second index
    t_sum.columns = pd.MultiIndex.from_arrays([t_sum.columns,['Sum'] * len(t_sum.columns)],names=('Var','Stats'))

    
    #t_mean.columns = pd.MultiIndex.from_arrays([t_mean.columns,['Bad Rate','Mean','Mean','Mean'] ],names=('Var','Stats'))
    t_mean.columns = pd.MultiIndex.from_arrays([t_mean.columns,['Bad Rate']+['Mean']*(len(t_mean.columns)-1)],names=('Var','Stats'))
    t_pct.columns = pd.MultiIndex.from_arrays([t_pct.columns,['Pct_of_Tot'] * len(t_pct.columns)], names=('Var','Stats'))
    
   
    
    if len(bps_var_col)==0:
        all_var=[t_sum,t_mean.set_index(t_sum.index),t_pct]
    else:
        all_var=[t_sum,t_mean.set_index(t_sum.index),t_bps,t_pct]
  
   
    
    t_all = pd.concat(all_var,axis=1,)
   
    t_all.sort_index(level=[0,1],axis=1,inplace=True,ascending=False)
    varnameall=['Total_Num']+varname
    final_table = t_all[varnameall]
    
    if (head<0) |(tail<0):
        raise ValueError("head or tail can not be less than 0")
    
    if (head==0) & (tail == 0):
    		#print score distribution
        if len(tran_var_col+bps_var_col+other_col) == 0:
            print('KS =',KS)
            print('IV=',IV)
            display(t_univ[['Total_Num','good_cnt','Bad_cnt','bad_rate','tot_pct','good_pct','bad_pct','cum_pct_gd','cum_pct_bd','WOE','bin_iv','KS']].style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))


        else:
            print('KS =',KS)
            print('IV=',IV)
            display(final_table.style.set_properties(**{
                           'align':"center",
                           'border-color':'blace',
                           'border-style' :'solid',
                           'border-width': '1px',
                           'border-collapse':'collapse'

        }).applymap(highlight_bps,subset=pd.IndexSlice[:, [col for col in zip(bps_var_col,['BPS']*len(bps_var_col))]]))   

    if head>0:
    		#print score distribution
        if len(tran_var_col+bps_var_col+other_col) == 0:
                print('KS =',KS)
                print('IV=',IV)
                display(t_univ[['Total_Num','good_cnt','Bad_cnt','bad_rate','tot_pct','good_pct','bad_pct','cum_pct_gd','cum_pct_bd','WOE','bin_iv','KS']].head(head).style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))
    
               
        else:
                print('KS =',KS)
                print('IV=',IV)
                display(final_table.head(head).style.set_properties(**{
                               'align':"center",
                               'border-color':'blace',
                               'border-style' :'solid',
                               'border-width': '1px',
                               'border-collapse':'collapse'

            })\
                    .applymap(highlight_bps,subset=pd.IndexSlice[:, [col for col in zip(bps_var_col,['BPS']*len(bps_var_col))]]))   

                
                
                

    if tail>0:
    		#print score distribution
        if len(tran_var_col+bps_var_col+other_col) == 0:
                print('KS =',KS)
                print('IV=',IV)
                display(t_univ[['Total_Num','good_cnt','Bad_cnt','bad_rate','tot_pct','good_pct','bad_pct','cum_pct_gd','cum_pct_bd','WOE','bin_iv','KS']].tail(tail).style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))
    
               
        else:
                print('KS =',KS)
                print('IV=',IV)
                display(final_table.tail(tail).style.set_properties(**{
                               'align':"center",
                               'border-color':'blace',
                               'border-style' :'solid',
                               'border-width': '1px',
                               'border-collapse':'collapse'

            })\
                    .applymap(highlight_bps,subset=pd.IndexSlice[:, [col for col in zip(bps_var_col,['BPS']*len(bps_var_col))]]))   

    #return final_table
    

    
