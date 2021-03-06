import pandas as pd
#master branch
import numpy as np

from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score, roc_curve,auc ,precision_recall_curve 

from IPython.display import display, HTML, Image
import matplotlib.pyplot as plt 


#from sklearn.externals.six import StringIO  
#from sklearn.tree import export_graphviz
#import pydotplus

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import sys

#sys.path.insert(0, '/Users/hemin/AnacondaProjects/Gitfolder/python_analytic_functions/')
sys.path.append("/Users/hemin/AnacondaProjects/Gitfolder/python_analytic_functions/")

from univ_fn import var_split

def GBtree_(X,y,n_tree=50,learning_rate=0.1, depth = 8,n_split =200,n_leaf = 30,n_var = 0.33,sub_samp=0.8,rand_seed = 1116,rand_size = 0.33, droplist=[]):
    
    X.drop(droplist,axis=1,inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=rand_size, random_state=rand_seed)
    
#     X_train.drop(droplist,axis=1,inplace=True)
#     X_test.drop(droplist,axis=1,inplace=True)
#     X.drop(droplist,axis=1,inplace=True)
    
    #bad rate in training and testing dataset
    train_good,train_bad = pd.value_counts(y_train)
    test_good,test_bad = pd.value_counts(y_test)
    
    train=[train_good,train_bad,train_good+train_bad,train_bad/(train_good+train_bad)]
    test =[test_good,test_bad,test_good+test_bad,test_bad/(test_good+test_bad)] 
    index=['Good','Bad','Total','Bad Rate']
    sum_data=pd.DataFrame({'Train':train,'Test':test},index=index)
    sum_data['Total']=sum_data['Train']+ sum_data['Test']
    sum_data_T=sum_data.T
    sum_data_T['Bad Rate']=(sum_data_T['Bad']/sum_data_T['Total']).map('{:,.2%}'.format)
    
    print(sum_data_T,'\n')

    dtree = GradientBoostingClassifier(n_estimators=n_tree,max_depth=depth,min_samples_split=n_split,
                min_samples_leaf=n_leaf,max_features=n_var,subsample=sub_samp,
                init=None,learning_rate=learning_rate ,random_state=1109                     
                                      )
    
    dtree.fit(X_train,y_train)
    
    y_train_pred = dtree.predict(X_train)
    y_test_pred  = dtree.predict(X_test)
    y_tot_pred  = dtree.predict(X)
    
 
    
    #probability
    a , b = dtree.classes_
    p_bad_train=pd.Series(dtree.predict_proba(X_train)[:,1] ,index=X_train.index).round(4)
    p_bad_test =pd.Series(dtree.predict_proba(X_test)[:,1]  ,index=X_test.index).round(4)
    p_bad_tot  =pd.Series(dtree.predict_proba(X)[:,1]       ,index=X.index).round(4)
   
    
   
    #Model Stats
    #AUC
    test_auc = roc_auc_score(y_test,p_bad_test)
    train_auc = roc_auc_score(y_train, p_bad_train)
    tot_auc = roc_auc_score(y, p_bad_tot)
    
    #KS,IV
    
    train_ks,train_iv,t =var_split(p_bad_train,y_train,bin_num= 10,showtb=False)   
    test_ks,test_iv,t1  =var_split(p_bad_test,y_test,bin_num= 10,showtb=False)
    tot_ks,tot_iv,t2    =var_split(p_bad_tot,y, bin_num= 10,showtb=False)

    index=['Train','Test','Total','% Change']
    sum_model_stats=pd.DataFrame({'KS':[train_ks,test_ks,tot_ks,(train_ks-test_ks)/train_ks] ,
                                  'IV':[train_iv,test_iv,tot_iv,(train_iv-test_iv)/train_iv],
                                  'AUC':[train_auc,test_auc,tot_auc,(train_auc-test_auc)/train_auc]},index=index)
    
    sum_model_stats.loc['% Change']= sum_model_stats.loc['% Change'].apply("{:.2%}".format)
    print(sum_model_stats,'\n')
    
    ####################################################
    ####precision and recall and GINI curve
    
    #precision and recall
    train_p,train_r,train_t = precision_recall_curve(y_train,p_bad_train)
    test_p,test_r,test_t = precision_recall_curve(y_test,p_bad_test)
    total_p,total_r,total_t = precision_recall_curve(y,p_bad_tot)

    bad_rate = np.mean(y)

    #graph gini
    
    gini_all_gpc=  t2.good_pct.str.rstrip('%').astype('float') / 100.0
    gini_all_bpc=  t2.bad_pct.str.rstrip('%').astype('float') / 100.0
    gini_all = pd.DataFrame({'good_pct':gini_all_gpc,'bad_pct':gini_all_bpc})
    gini_all[['cum_pct_bad','cum_pct_good']]=gini_all[::-1].cumsum() - 1
    
    gini_test_gpc=  t1.good_pct.str.rstrip('%').astype('float') / 100.0
    gini_test_bpc=  t1.bad_pct.str.rstrip('%').astype('float') / 100.0
    gini_test = pd.DataFrame({'good_pct':gini_test_gpc,'bad_pct':gini_test_bpc})
    gini_test[['cum_pct_bad','cum_pct_good']]=gini_test[::-1].cumsum() - 1
    
    gini_train_gpc=  t.good_pct.str.rstrip('%').astype('float') / 100.0
    gini_train_bpc=  t.bad_pct.str.rstrip('%').astype('float') / 100.0
    gini_train = pd.DataFrame({'good_pct':gini_train_gpc,'bad_pct':gini_train_bpc})
    gini_train[['cum_pct_bad','cum_pct_good']]=gini_train[::-1].cumsum() - 1
    
    
    
    plt.figure(figsize=(10, 4))
   
    plt.subplot(121)
    plt.step(train_r, train_p, color='r', alpha=0.9, where='post', label='Train')
    plt.step(test_r, test_p, color='y', alpha=0.9, where='post', label='Test')
    plt.step(total_r, total_p, color='b', alpha=0.9, where='post', label='Total')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.axhline(y= bad_rate, color='r', linestyle='-',label='Random')
    plt.title('Precision-Recall Curve: bad_rate={0:0.2f}'.format(bad_rate))
    plt.legend()
   


    plt.subplot(122)
    plt.title('GINI Curve')
    plt.plot(gini_all.cum_pct_good,gini_all.cum_pct_bad,'b-',label='Total')
    plt.plot(gini_train.cum_pct_good,gini_train.cum_pct_bad,'r-',label='Train') 
    plt.plot(gini_test.cum_pct_good,gini_test.cum_pct_bad,'y-',label='Test')
    
    plt.plot([0,1], [0,1], 'k--')
    plt.legend(loc='lower right')
    plt.xlabel("% Cumulative Good", fontweight='bold')
    plt.ylabel("% Cumulative Bad", fontweight='bold')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.40)
    plt.plot()
    plt.show()

   
 ################################################   
    
    num_bin=10
    
    print('\n')
    print("Score distribution for Building")
    if num_bin>40:
        display(t.tail(20).style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))
    else:
        display(t.style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))
    
    print('\n')
    print("Score distribution for Testing")
    
    if num_bin>40:
        display(t1.tail(20).style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))
    else:
        display(t1.style.bar(subset=['WOE'], align='mid', color=['#d65f5f', '#5fba7d']))
   
    #importance
    varname=X_train.columns.tolist()
    imp = dtree.feature_importances_

  
    
    importance = pd.DataFrame({'Varname':varname,'Importance':imp}).sort_values(by='Importance',ascending = False)
    importance['Importance']=importance['Importance'].map('{:,.3f}'.format)
    print('\n')
    print("Feature ranking(Top 20):",'\n')
    print(importance[['Varname','Importance']].head(20))
    
    
    print("Training Set",'\n')                              
    print(classification_report(y_train,y_train_pred))      
    print('\n')                                             
    print(confusion_matrix(y_train,y_train_pred),'\n')      
                                                            
    print("Testing Set")                                    
    print(classification_report(y_test,y_test_pred),'\n')   
    print(confusion_matrix(y_test,y_test_pred),'\n')        
                                                            
                                                             
    print("Total")                                          
    print(classification_report(y,y_tot_pred),'\n')         
    print(confusion_matrix(y,y_tot_pred),'\n')              
 
    
     
     
    #s1=pd.DataFrame({'SLR_ID':X_train.SLR_ID,'perf_flag':y_train, 'flag_test': [0]*len(y_train) ,'p_prob':p_bad_train})
    #s2=pd.DataFrame({'SLR_ID':X_test.SLR_ID,'perf_flag':y_test, 'flag_test': [1]*len(y_test),'p_prob':p_bad_test})
    
    ####combine data
    s1=pd.DataFrame({'perf_y':y_train, 'flag_test': [0]*len(y_train) ,'p_prob':p_bad_train})
    s2=pd.DataFrame({'perf_y':y_test, 'flag_test': [1]*len(y_test),'p_prob':p_bad_test})
    
    
    s_prob=pd.concat([s1,s2],axis=0)
    
    
    return dtree,s_prob
