from pandas.core.dtypes.missing import isnull
from share_sys_module import *


class ScoreRank():
    '''

    ScoreRank(datin, bin_num,class_var,perf_var_col,tran_var_col,bps_var_col,other_col)


    This function is used to create score cutoffs, score distributions in terms of unit bad
    and other performance metrics such as Net Loss, Claims etc. , GINI Curve, Precision/Recall Curve
    ,KS,IV,Univariate Tables and Strategy tables.

    python: 3.6.5
    pandas: 0.23.1

    This function is to replace score_dist function
    
    Mike Min 08/2019
    
    '''

    
    
    
    def __init__(self,datin,bin_num,class_var,perf_var_col,tran_var_col,bps_var_col,other_col):
    
    
        '''
        
        datin: Input dataframe
        bin_num: how many bins for score. if class_var is not continous , it will be ignored 
        class_var: score or any variables you would like to see the performance metrics by groups
        perf_var_col: unit bad
        tran_var_col: GMV
        bos_var_col: a list of net losses, claims etc.
        other_col: such as suspension flag 
        
        example:
        s = ScoreRank(test100k,10,'bocv2_score',['perf_bad_30d'],['perf_gmv_30d'],['perf_amt_open_inr_30d', 'perf_rvi_net_loss_30d','perf_amt_esc_clm_30d'],['flag_cur_susp','perf_res_30d','perf_susp_30d','perf_susp_hold_30d'])
        
        Methods:
        s.showUnivTable()
        s.showDistTable()
        s.showStrTable()
        s.GraphTable(den,num,gtype='',title='' )
        
        '''
        
        self.__datin = datin.copy()
        self.__bin_num = bin_num
        self.__class_var= datin[class_var]
        self.__class_var_raw = class_var
        self.__perf_var_col = perf_var_col
        self.__tran_var_col = tran_var_col
        self.__bps_var_col = bps_var_col
        self.__other_col = other_col
        self.__varname = perf_var_col+tran_var_col+bps_var_col+other_col
        self.__meanvar = tran_var_col+bps_var_col  #for the format of means 
        
        
        if self.__bin_num<= 0 | isnull(self.__bin_num):
            raise ValueError("Bin_num cannot be zeor or missing")
    
        #if not is_numeric_dtype(self.__class_var):
            #raise ValueError('input variable must be numeric')
        
        if len(class_var)==0 | len(perf_var_col) == 0:
                raise ValueError("Class var or perf var cannot be zeor or missing")
    
    def _bincut(self,numbin):
        
        #if not np.issubdtype(x.dtype, np.number): (does not work for category)
            #raise ValueError('input variable must be numeric')

        quantiles=[n/numbin for n in range(0,numbin+1)]
        #print(quantiles)
        ###bins for interval
        bin_all = algos.quantile(self.__class_var, quantiles).round(4)

        #bins for grouping
        bins = bin_all[1:len(bin_all)]

        bin_all[0]=(bin_all[0]-0.001).round(4)
        if self.__class_var.isnull().any():
            bin_all=np.append(-999999999,bin_all)

        uni_bin_all=algos.unique(bin_all)
        uni_bins = algos.unique(bins)
        #print(bins)
        #print(uni_bins)
        ids=_ensure_int64(uni_bins.searchsorted(self.__class_var.round(4), side='left'))

        if ids[ids==len(uni_bins)].any() | isnull(self.__class_var).any() :  
            ids=ids+1
            ids[ids == (len(uni_bins)+1)] = 0


        labels = IntervalIndex.from_breaks(uni_bin_all, closed='right')



        result = algos.take_nd(labels, ids)
        #print(pd.Series.count(result))

        return result
    
    
    
    def bincutall(self,numbin = 0):
        '''
        Create score bins
        
        
        '''
        if numbin == 0:
            numbin = self.__bin_num
        
        if is_numeric_dtype(self.__class_var):
            if self.__class_var.nunique() <= 10:
                self.__class_var = self.__class_var.fillna('missing')
                fac = self.__class_var
            else:
                #fac=self._bincut(numbin)
                fac = pd.qcut(self.__class_var,numbin,duplicates='drop').cat.add_categories("Missing").fillna('Missing')

        else:
            if isinstance(self.__class_var.dtype,CategoricalDtype):
                self.__class_var = self.__class_var.cat.add_categories("Missing").fillna('Missing')
            else:
                self.__class_var = self.__class_var.fillna('missing')
            fac = self.__class_var

        return fac
    
    
    def createTable(self):
        '''
        return variable table, univariate table ,KS and IV
        final_table,t_univ,KS,IV = s.createTable()
        
        '''
        
        self.__datin['bins']=self.bincutall()
        

        t=self.__datin.groupby(['bins'])
      

        #SUM
        t_sum1=pd.concat([t.size(),t[self.__varname].sum()],axis=1)
        t_sum1.rename(columns={0:'Total_Num'},inplace=True)

        #total number
        sum_all=self.__datin[self.__varname].sum()
        sum_tot=pd.Series([self.__datin.shape[0]],index=['Total_Num']).append(sum_all)
        sum_tot1=sum_tot.copy()

        t_sum1[self.__class_var_raw]=t_sum1.index
        sum_tot1[self.__class_var_raw]='Total'


        t_sum = t_sum1.append(sum_tot1, ignore_index=True)
        t_sum.set_index(self.__class_var_raw, inplace=True)

        ###### Univariate and calculate KS and IV


        t_univ = t_sum.iloc[:,0:2]
        #t_univ.columns.values[1] = 'Bad_cnt' #where change var name in the original Dataframe
        t_univ.rename(columns={ t_univ.columns[1]: "Bad_cnt" },inplace=True)



        t_univ['good_cnt'] = t_univ['Total_Num'] - t_univ['Bad_cnt']


        ## in case log(0) or divided by 0; for WOE and IV calucalation 
        t_univ.Bad_cnt[t_univ['Bad_cnt']==0]=0.00001


        t_univ[['tot_pct','bad_pct','good_pct']] = t_univ.div(t_univ.loc['Total',:],axis=1)

        t_univ[['cum_tot','cum_bad','cum_good','cum_pct_tot','cum_pct_bd','cum_pct_gd']] = t_univ.cumsum().round(3)
        
        ## reversed cumulative
        ##t_univ[['cum_tot','cum_bad','cum_good','cum_pct_tot','cum_pct_bd','cum_pct_gd']] = t_univ[::-1].cumsum()[::-1].round(3)
        
        
        t_univ.loc['Total',['cum_tot','cum_bad','cum_good','cum_pct_tot','cum_pct_bd','cum_pct_gd']] = 0


        t_univ['WOE']= np.log(t_univ['good_pct']/t_univ['bad_pct']).round(3)

        t_univ['bin_iv'] = ((t_univ['good_pct'] - t_univ['bad_pct'])*t_univ['WOE']).round(3)
        t_univ['KS'] = abs((t_univ.cum_pct_gd - t_univ.cum_pct_bd)).round(3)*100

        t_univ['bad_rate']=(t_univ['Bad_cnt']/t_univ['Total_Num'])
        t_univ[['bad_rate','tot_pct','good_pct','bad_pct']]=t_univ[['bad_rate','tot_pct','good_pct','bad_pct']].apply(lambda x : x.map('{:,.2%}'.format),axis=1)
        IV =t_univ.bin_iv.sum()
        t_univ.loc['Total',['bin_iv']] = IV
        KS=abs((t_univ.cum_pct_gd - t_univ.cum_pct_bd)).max().round(4)*100



        ###################### Variable summary
        t_mean=t[self.__varname].mean().round(2)
        mean_all=self.__datin[self.__varname].mean().round(2)
        t_mean = t_mean.append(mean_all, ignore_index=True)


        t_mean[self.__meanvar]=t_mean[self.__meanvar].apply(lambda x : x.map('{:,.0f}'.format),axis=1)



        #t_mean['perf_flag_x'] = t_mean['perf_flag_x'].map('{:,.2%}'.format)
        t_mean[self.__perf_var_col] = t_mean[self.__perf_var_col].apply(lambda x : x.map('{:,.2%}'.format),axis=1)

        #print(t_mean[meanvar].describe())


        if len(self.__bps_var_col)>0:
            t_bps=t_sum[self.__bps_var_col].div(t_sum.iloc[:,2], axis=0).mul(10000).apply(lambda x : x.map('{:,.0f}'.format),axis=1)
            t_bps.columns = pd.MultiIndex.from_arrays([t_bps.columns,['BPS'] * len(t_bps.columns)],names=('Var','Stats'))

        t_pct=t_sum[['Total_Num']+self.__varname].div(sum_tot,axis=1).apply(lambda x : x.map('{:,.2%}'.format),axis=1)
        t_sum=t_sum.apply(lambda x : x.map('{:,.0f}'.format),axis=1)


        # #add second index
        t_sum.columns = pd.MultiIndex.from_arrays([t_sum.columns,['Sum'] * len(t_sum.columns)],names=('Var','Stats'))


        #t_mean.columns = pd.MultiIndex.from_arrays([t_mean.columns,['Bad Rate','Mean','Mean','Mean'] ],names=('Var','Stats'))
        t_mean.columns = pd.MultiIndex.from_arrays([t_mean.columns,['Bad Rate']+['Mean']*(len(t_mean.columns)-1)],names=('Var','Stats'))
        t_pct.columns = pd.MultiIndex.from_arrays([t_pct.columns,['Pct_of_Tot'] * len(t_pct.columns)], names=('Var','Stats'))



        if len(self.__bps_var_col)==0:
            all_var=[t_sum,t_mean.set_index(t_sum.index),t_pct]
        else:
            all_var=[t_sum,t_mean.set_index(t_sum.index),t_bps,t_pct]



        t_all = pd.concat(all_var,axis=1,)

        t_all.sort_index(level=[0,1],axis=1,inplace=True,ascending=False)
        varnameall=['Total_Num']+self.__varname
        final_table = t_all[varnameall]
        
        
        
        return final_table,t_univ,KS,IV
     
    def showUnivTable(self,head=0, tail=0):
        
        '''
        show Univariate table.
        s.showUnivTable()
        default is to output all bins
        s.showUnivTable(head=10,tail= 0)
        
        '''
        
        
    #print univariate table
        a,t_univ,KS,IV = self.createTable()
    
        print('KS =',KS)
        print('IV=',IV)
    
        if (head < 0) | (tail<0):
            raise ValueError("head or tail can not be less than 0")
    
        if (head == 0) and (tail==0):
            display(t_univ[['Total_Num','good_cnt','Bad_cnt','bad_rate','tot_pct','good_pct','bad_pct','cum_pct_gd','cum_pct_bd','WOE','bin_iv','KS']].style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))
    
        if (tail > 0):
            display(t_univ[['Total_Num','good_cnt','Bad_cnt','bad_rate','tot_pct','good_pct','bad_pct','cum_pct_gd','cum_pct_bd','WOE','bin_iv','KS']].tail(tail).style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))
    
        if (head > 0 ):
            display(t_univ[['Total_Num','good_cnt','Bad_cnt','bad_rate','tot_pct','good_pct','bad_pct','cum_pct_gd','cum_pct_bd','WOE','bin_iv','KS']].head(head).style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))
    
    
    
    
    def _showDistTable(self,head=0, tail=0): 
        '''
        Old one
        show variable distributions by score bins.
        s.showDistTable()
        default is to output all bins
        s.showDistTable(head=10,tail= 0)
        
        
        '''
        
        
        if (head < 0) | (tail<0):
            raise ValueError("head or tail can not be less than 0")
        if len(self.__tran_var_col+self.__bps_var_col+self.__other_col) == 0:
            raise ValueError("No analytic variables are provided, Please use showUnivTable method")
            
        final_table,b,KS,IV = self.createTable()
        
        print('KS =',KS)
        print('IV=',IV)
        
        #display(final_table.tail(121).style.applymap(self.highlight_bps,subset=pd.IndexSlice[:, [col for col in zip(self.__bps_var_col,   ['BPS']*len(self.__bps_var_col))]]))

        if (head == 0) and (tail==0):
            display(final_table.style.set_properties(**{
                               'align':"center",
                               'border-color':'blace',
                               'border-style' :'solid',
                               'border-width': '1px',
                               'border-collapse':'collapse'

            })\
                    .applymap(self._highlight_bps,subset=pd.IndexSlice[:, [col for col in zip(self.__bps_var_col,['BPS']*len(self.__bps_var_col))]]))   

        if  tail>0 :
            display(final_table.tail(tail).style.set_properties(**{
                               'align':"center",
                               'border-color':'blace',
                               'border-style' :'solid',
                               'border-width': '1px',
                               'border-collapse':'collapse'

            })\
                    .applymap(self._highlight_bps,subset=pd.IndexSlice[:, [col for col in zip(self.__bps_var_col,['BPS']*len(self.__bps_var_col))]]))   

        if  head > 0 :
            display(final_table.head(head).style.set_properties(**{
                               'align':"center",
                               'border-color':'blace',
                               'border-style' :'solid',
                               'border-width': '1px',
                               'border-collapse':'collapse'

            })\
                    .applymap(self._highlight_bps,subset=pd.IndexSlice[:, [col for col in zip(self.__bps_var_col,['BPS']*len(self.__bps_var_col))]]))   


    
    def _highlight_bps(self,s):
        color = 'yellow'
        return 'background-color: %s' % color
    
    
    
    def GraphTable(self,den,num,gtype='',title='' ):
        
        '''
        graph precision/recall curve, GINI curve for unit bad and other perf variables

        if do not specify gtype, it will output both gini and precision/recall

        Unit bad:
        s.GraphTable('','perf_bad_30d',title = 'Perf_bad_30d',gtype = 'GINI')

        perf vars:

        s.GraphTable('perf_gmv_30d','perf_amt_open_inr_30d',title = 'INR rate')


        '''
    
        if len(self.__class_var_raw)==0 | len(num) == 0:
            raise ValueError("Spliter or numerator cannot be zeor or missing")

        

        #copy input data
        if len(den) == 0:
            __df = self.__datin[[self.__class_var_raw,num]].copy()
        else:
            __df = self.__datin[[self.__class_var_raw,den,num]].copy()


    

        # group by all scores
        if len(den) == 0:
            __df['Total'] = np.ones(len(self.__class_var))
            bad_rate = np.mean(self.__datin[num])

            s=__df.groupby([self.__class_var_raw])['Total',num].sum().fillna(0)
            s.rename(columns={num:'Sum_Bad'},inplace=True)
        else:
            s=__df.groupby([self.__class_var_raw])[den,num].sum().fillna(0)
            s.rename(columns={den:'Total',num:'Sum_Bad'},inplace=True)
            
            bad_rate = __df[num].sum()/__df[den].sum()

        print('Bad_rate: ',bad_rate)    
        s['Sum_Good'] = s['Total']-s['Sum_Bad']
        sum_tot_bad = np.sum(s.Sum_Bad)
        sum_tot_good = np.sum(s.Sum_Good)

        s['good_pct'] = s['Sum_Good']/sum_tot_good
        s['bad_pct'] = s['Sum_Bad']/sum_tot_bad

        s.sort_index(inplace=True,ascending =False)

        s[['cum_tot','cum_bad','cum_good','cum_pct_gd','cum_pct_bd']] = s.cumsum()

        s['precision']=s['cum_bad']/s['cum_tot']
        s['recall'] = s['cum_pct_bd']

        #display(s.head(20))
        

        #calculate GINI
        zeros = pd.DataFrame([[0,0]],columns=['cum_pct_gd','cum_pct_bd'])
        gini = zeros.append(s[['cum_pct_gd','cum_pct_bd']],ignore_index=True)

        #### plot GINI and PR
        
        if gtype == '':
            plt.figure(figsize=(14, 6))


            plt.subplot(121)

            plt.step(s.recall, s.precision, color='b', alpha=0.9, where='post', label='Total: '+ self.__class_var_raw)

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            #plt.ylim([0.0, 1.05])
            plt.ylim([0.0, 25*bad_rate])
            plt.xlim([0.0, 1.05])
            plt.axhline(y= bad_rate, color='r', linestyle='-',label='Random')
            plt.title('Precision-Recall Curve: ' + title +'\n bad_rate={0:0.2f}'.format(bad_rate))

            plt.legend()



            plt.subplot(122)
            plt.title('GINI Curve: '+title)
            plt.plot(gini.cum_pct_gd,gini.cum_pct_bd,'b-',label='Total: ' + self.__class_var_raw)


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
            #print("Precision/Recall:",'\n')
            plt.plot(s.recall,s.precision,'y')
            plt.axhline(y= bad_rate, color='r', linestyle='-',label='Random')
            plt.title('Precision-Recall Curve: ' + title +'\n bad_rate={0:0.2f}'.format(bad_rate))
            
            

        #plot gini
        if gtype == 'GINI':

            print('\n')
            
            plt.title('GINI Curve: '+title)
            plt.plot(gini.cum_pct_gd,gini.cum_pct_bd,'b-',label='Total: ' + self.__class_var_raw)


            plt.plot([0,1], [0,1], 'k--',label='Random')
            plt.legend(loc='lower right')
            plt.xlabel("% Cumulative Good", fontweight='bold')
            plt.ylabel("% Cumulative Bad", fontweight='bold')
    
    
    
    def _set_total(self,gdat):
        '''
        Add Total to a categorical index 
        qcut creates a categorical index which is not able to append total using 
        str_b=strdat.append(strdat.sum().rename('Total'),ignore_index=False)
        or
        df.loc['Total']= df.sum()
        '''
        
        
        
        total_sum = gdat.sum()
        gdat['score']=gdat.index
        total_sum['score'] = 'Total'
        gdat_ = gdat.append(total_sum,ignore_index = True)
        gdat_.set_index('score',inplace = True)
        return gdat_

    
    
    def _strVar(self,strdatin):
        '''
        create percentage variables and cumulative variables for strategy table
        
        '''
        
        strdat = strdatin.copy()
        var_pct=[]
        for i in strdat.columns.tolist():
            var_pct.append('pct_'+i)


        strdat[var_pct] = strdat.div(strdat.sum(),axis=1)

        var_cum=[]
        for i in strdat.columns.tolist():
            var_cum.append('cum_'+i)



        strdat[var_cum]=strdat[::-1].cumsum()[::-1].round(3)

        
        str_b = self._set_total(strdat)
        #str_b=strdat.append(strdat.sum().rename('Total'),ignore_index=False)
        #str_b[self.__class_var_raw] = str_b.index
        #str_b.set_index(self.__class_var_raw,inplace=True)
        
        
        
        
        #reset the sum of cum var to zero
        str_b.loc['Total',var_cum] = 0

        return str_b
        
        
    def showStrTable(self,binnum=0):
        '''
        Cumulative percentage of unit bad and perf vars.
        support re-define bins
        
        '''
        
        
        
        if binnum==0:
            binnum = self.__bin_num
        
        self.__datin['bins']=self.bincutall(numbin = binnum)
        
        g = self.__datin.groupby('bins')
        
        
        
        #unit bad
        str_a = g.agg({self.__perf_var_col[0]:['count','sum']})
        str_a.columns = str_a.columns.droplevel()
        
        str_a.rename({'count':'tot_cnt','sum':'bad_cnt'},inplace=True,axis=1)
        str_a['good_cnt'] = str_a.tot_cnt - str_a.bad_cnt
        
        str_bi = self._strVar(str_a)
        
        str_bi['bad_rate'] = str_bi.bad_cnt/str_bi.tot_cnt
        str_bi['cum_bad_rate'] =  str_bi.cum_bad_cnt/str_bi.cum_tot_cnt
        
        str_bi[['bad_rate','pct_tot_cnt','pct_good_cnt','pct_bad_cnt','cum_bad_rate','cum_pct_tot_cnt','cum_pct_good_cnt','cum_pct_bad_cnt']]= \
        str_bi[['bad_rate','pct_tot_cnt','pct_good_cnt','pct_bad_cnt','cum_bad_rate','cum_pct_tot_cnt','cum_pct_good_cnt','cum_pct_bad_cnt']].apply(lambda x : x.map('{:,.2%}'.format),axis=1)
        
        varOutput = ['tot_cnt','bad_cnt','pct_tot_cnt','pct_bad_cnt','bad_rate','cum_pct_tot_cnt','cum_pct_bad_cnt','cum_bad_rate','cum_tot_cnt','cum_bad_cnt']
        
        display(str_bi[varOutput].style.applymap(self._highlight_bps,subset=['bad_rate','cum_bad_rate']))
        
        #Perf Var
        
        perfvarlist = self.__tran_var_col+self.__bps_var_col +self.__other_col 
        
        if len(perfvarlist) != 0:
           
            str_ = g[perfvarlist].sum()
            str_perf = self._strVar(str_)
          
            
            #calculate cumulative rate for perf vars
            cum_var = ['cum_'+ s for s in self.__bps_var_col]
            varOutput3_rate = ['rat_'+ s for s in cum_var]
            
            
            str_perf_rate = str_perf[cum_var].div(str_perf['cum_'+str(self.__tran_var_col[0])],axis=0)\
                            .apply(lambda x : x.map('{:,.2%}'.format),axis=1)
            
            str_perf_rate.columns = varOutput3_rate
            
            #output var list
            
            varOutput_cumpct = ['cum_pct_'+ s for s in perfvarlist]
            
            varOutput2_pct = self.__tran_var_col+['pct_'+str(self.__tran_var_col[0])] 
            
            var_test = ['pct_'+str(self.__tran_var_col[0])] 
            
            varOutput3_raw = self.__bps_var_col +self.__other_col
            
            str_var = pd.concat([str_perf,str_perf_rate],axis=1)
            final_output_var = varOutput2_pct+ varOutput_cumpct+varOutput3_rate+varOutput3_raw
            
            str_var[varOutput_cumpct] = str_var[varOutput_cumpct].apply(lambda x : x.map('{:,.2%}'.format),axis=1)
            
            
            
            str_var[self.__tran_var_col+self.__bps_var_col] = str_var[self.__tran_var_col+self.__bps_var_col]\
                                .apply(lambda x : x.map('{:,.0f}'.format),axis=1)
            
            
            str_var[var_test] = str_var[var_test]\
                                .apply(lambda x : x.map('{:,.2%}'.format),axis = 1)
            
                    
            
            display(str_var[final_output_var].style.set_properties(**{
                               'align':"center",
                               'border-color':'blace',
                               'border-style' :'solid',
                               'border-width': '1px',
                               'border-collapse':'collapse'

            })\
            .applymap(self._highlight_bps,subset=varOutput3_rate)   
                  
            )

            

            
            
            
            
    def _showFmt(self,showTable,head=0,tail=0):
        
        if (head < 0) | (tail<0):
            raise ValueError("head or tail can not be less than 0")
        if len(self.__tran_var_col+self.__bps_var_col+self.__other_col) == 0:
            raise ValueError("No analytic variables are provided, Please use showUnivTable method")
            
        
        
        
        
        #display(final_table.tail(121).style.applymap(self.highlight_bps,subset=pd.IndexSlice[:, [col for col in zip(self.__bps_var_col,   ['BPS']*len(self.__bps_var_col))]]))

        if (head == 0) and (tail==0):
            display(showTable.style.set_properties(**{
                               'align':"center",
                               'border-color':'blace',
                               'border-style' :'solid',
                               'border-width': '1px',
                               'border-collapse':'collapse'

            })\
                    .applymap(self._highlight_bps,subset=pd.IndexSlice[:, [col for col in zip(self.__bps_var_col,['BPS']*len(self.__bps_var_col))]]))   

        if  tail>0 :
            display(showTable.tail(tail).style.set_properties(**{
                               'align':"center",
                               'border-color':'blace',
                               'border-style' :'solid',
                               'border-width': '1px',
                               'border-collapse':'collapse'

            })\
                    .applymap(self._highlight_bps,subset=pd.IndexSlice[:, [col for col in zip(self.__bps_var_col,['BPS']*len(self.__bps_var_col))]]))   

        if  head > 0 :
            display(showTable.head(head).style.set_properties(**{
                               'align':"center",
                               'border-color':'blace',
                               'border-style' :'solid',
                               'border-width': '1px',
                               'border-collapse':'collapse'

            })\
                    .applymap(self._highlight_bps,subset=pd.IndexSlice[:, [col for col in zip(self.__bps_var_col,['BPS']*len(self.__bps_var_col))]]))   


    def showDistTable(self,head=0,tail=0):
        
        '''
        show variable distributions by score bins.
        s.showDistTable()
        default is to output all bins
        s.showDistTable(head=10,tail= 0)
        
        
        '''
        
        final_table,b,KS,IV = self.createTable() 
        
        print('KS =',KS)
        print('IV=',IV)
        
        self._showFmt(final_table,head,tail)
        
        
        
