#draw tree; server version may not have package
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score, roc_curve,auc
from IPython.display import display, HTML, Image

from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier





import sys

sys.path.insert(0, '/Users/hemin/AnacondaProjects/grpc_ds_python_modules-master_py36/')
#sys.path.insert(0, '/home/hemin/py_project/code_lib/grpc_ds_python_modules-master_py36/')
from ksiv import ks_iv_single,  ks_iv_array, univ_single,univ_array,eq_bin, wp 

def GBtree_(X,y,n_tree=50,learning_rate=0.1, depth = 8,n_split =200,n_leaf = 30,n_var = 0.33,sub_samp=0.8, droplist=[]):
    
    X_=X.copy()
    y_=y.copy()
    X_.drop(droplist,axis=1,inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.33, random_state=32)
    
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
                init=None,learning_rate=0.1 ,random_state=1109                     
                                      )
    
    dtree.fit(X_train,y_train)
    
    y_train_pred = dtree.predict(X_train)
    y_test_pred  = dtree.predict(X_test)
    y_tot_pred  = dtree.predict(X_)
    
    
    print("Training Set",'\n')
    print(classification_report(y_train,y_train_pred))
    print('\n')
    print(confusion_matrix(y_train,y_train_pred),'\n')
    
    print("Testing Set")
    print(classification_report(y_test,y_test_pred),'\n')
    print(confusion_matrix(y_test,y_test_pred),'\n')
    
    
    print("Total")
    print(classification_report(y,y_tot_pred),'\n')
    print(confusion_matrix(y_,y_tot_pred),'\n')
    
    #probability
    a , b = dtree.classes_
    p_bad_train=dtree.predict_proba(X_train)[:,1]
    p_bad_test=dtree.predict_proba(X_test)[:,1]
    p_bad_tot=dtree.predict_proba(X_)[:,1]
   
    #Model Stats
    #AUC
    test_auc = roc_auc_score(y_test,p_bad_test)
    train_auc = roc_auc_score(y_train, p_bad_train)
    tot_auc = roc_auc_score(y_, p_bad_tot)
    
    #KS,IV
    num_bin=10
    train_ks,train_iv,c,d=ks_iv_single(p_bad_train,y_train, wgt=None , n_bin=num_bin, var_type='continuous', keepna=True)
    test_ks,test_iv,c1,d1=ks_iv_single(p_bad_test,y_test, wgt=None , n_bin=num_bin, var_type='continuous', keepna=True)
    tot_ks,tot_iv,c2,d2=ks_iv_single(p_bad_tot,y_, wgt=None , n_bin=10, var_type='continuous', keepna=True)
    
#     print("Train_KS = {:,.3f} \nTest_KS = {:,.3f} \nTot_KS={:,.3f} \nKS Diff between Train and Test:{:,.2%}".format(train_ks, test_ks,tot_ks, (train_ks - test_ks)/train_ks))
#     print("Train_IV = {:,.3f} \nTest_IV = {:,.3f} \nTot_IV={:,.3f} \nIV Diff between Train and Test:{:,.2%}".format(train_iv, test_iv,tot_iv, (train_iv - test_iv)/train_iv))
#     print("Train_auc = {:,.3f} \nTest_auc = {:,.3f} \nTot_auc={:,.3f} \nAUC Diff between Train and Test:{:,.2%}".format(train_auc, test_auc,tot_auc, (train_auc - test_auc)/train_auc))
    
    index=['Train','Test','Total','% Change']
    sum_model_stats=pd.DataFrame({'KS':[train_ks,test_ks,tot_ks,(train_ks-test_ks)/train_ks] ,
                                  'IV':[train_iv,test_iv,tot_iv,(train_iv-test_iv)/train_iv],
                                  'AUC':[train_auc,test_auc,tot_auc,(train_auc-test_auc)/train_auc]},index=index)
    
    sum_model_stats.loc['% Change']= sum_model_stats.loc['% Change'].apply("{:.2%}".format)
    print(sum_model_stats,'\n')
    print('\n')
    print("Score distribution for Building")
    if num_bin>40:
        display(d.tail(10))
    else:
        display(d)
    
    print('\n')
    print("Score distribution for Testing")
    
    if num_bin>40:
        display(d1.tail(10))
    else:
        display(d1)
    
    #importance
    varname=X_train.columns.tolist()
    imp = dtree.feature_importances_

  
    
    importance = pd.DataFrame({'Varname':varname,'Importance':imp}).sort_values(by='Importance',ascending = False)
    importance['Importance']=importance['Importance'].map('{:,.3f}'.format)
    print('\n')
    print("Feature ranking(Top 20):",'\n')
    print(importance[['Varname','Importance']].head(20))
    
   
    #s1=pd.DataFrame({'SLR_ID':X_train.SLR_ID,'perf_flag':y_train, 'flag_test': [0]*len(y_train) ,'p_prob':p_bad_train})
    #s2=pd.DataFrame({'SLR_ID':X_test.SLR_ID,'perf_flag':y_test, 'flag_test': [1]*len(y_test),'p_prob':p_bad_test})
    
    s1=pd.DataFrame({'perf_y':y_train, 'flag_test': [0]*len(y_train) ,'p_prob':p_bad_train})
    s2=pd.DataFrame({'perf_y':y_test, 'flag_test': [1]*len(y_test),'p_prob':p_bad_test})
    
    
    s_prob=pd.concat([s1,s2],axis=0)
    
    return dtree,s_prob

def highlight_bps(s):
        color = 'yellow'
        return 'background-color: %s' % color  
    
    


def dtree_(X,y,depth = 3,n_split = 2,n_leaf = 1,n_var = 0.33,draw_Tree=False , droplist=[] ):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=32)
    
    
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


    
    dtree = DecisionTreeClassifier(max_depth=depth,min_samples_split=n_split , min_samples_leaf=n_leaf,max_features=None )
    dtree.fit(X_train.drop(droplist,axis=1),y_train)
    
    y_train_pred = dtree.predict(X_train)
    y_test_pred  = dtree.predict(X_test)
    y_tot_pred  = dtree.predict(X)
    
    
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
    
    #probability
    a , b = dtree.classes_
    p_bad_train=dtree.predict_proba(X_train)[:,1]
    p_bad_test=dtree.predict_proba(X_test)[:,1]
    p_bad_tot=dtree.predict_proba(X)[:,1]
   
    #Model Stats
    #AUC
    test_auc = roc_auc_score(y_test,p_bad_test)
    train_auc = roc_auc_score(y_train, p_bad_train)
    tot_auc = roc_auc_score(y, p_bad_tot)
    
    #KS,IV
    train_ks,train_iv,c,d=ks_iv_single(p_bad_train,y_train, wgt=None , n_bin=10, var_type='continuous', keepna=True)
    test_ks,test_iv,c1,d1=ks_iv_single(p_bad_test,y_test, wgt=None , n_bin=10, var_type='continuous', keepna=True)
    tot_ks,tot_iv,c2,d2=ks_iv_single(p_bad_tot,y, wgt=None , n_bin=10, var_type='continuous', keepna=True)
    
#     print("Train_KS = {:,.3f} \nTest_KS = {:,.3f} \nTot_KS={:,.3f} \nKS Diff between Train and Test:{:,.2%}".format(train_ks, test_ks,tot_ks, (train_ks - test_ks)/train_ks))
#     print("Train_IV = {:,.3f} \nTest_IV = {:,.3f} \nTot_IV={:,.3f} \nIV Diff between Train and Test:{:,.2%}".format(train_iv, test_iv,tot_iv, (train_iv - test_iv)/train_iv))
#     print("Train_auc = {:,.3f} \nTest_auc = {:,.3f} \nTot_auc={:,.3f} \nAUC Diff between Train and Test:{:,.2%}".format(train_auc, test_auc,tot_auc, (train_auc - test_auc)/train_auc))
    
    index=['Train','Test','Total','% Change']
    sum_model_stats=pd.DataFrame({'KS':[train_ks,test_ks,tot_ks,(train_ks-test_ks)/train_ks] ,
                                  'IV':[train_iv,test_iv,tot_iv,(train_iv-test_iv)/train_iv],
                                  'AUC':[train_auc,test_auc,tot_auc,(train_auc-test_auc)/train_auc]},index=index)
    
    sum_model_stats.loc['% Change']= sum_model_stats.loc['% Change'].apply("{:.2%}".format)
    print(sum_model_stats,'\n')
    print('\n')
    print("Score distribution for Building")
    display(d)
    
    print('\n')
    print("Score distribution for Testing")
    display(d1)
    
    #importance
    varname=X_train.columns.tolist()
    imp = dtree.feature_importances_

    # indices = np.argsort(imp)[::-1]
    # for f in range(3):
    # print( (f+1,varname[indices[f]], indices[f],"{:,.2f}".format(imp[indices[f]])))
    
    importance = pd.DataFrame({'Varname':varname,'Importance':imp}).sort_values(by='Importance',ascending = False)
    importance['Importance']=importance['Importance'].map('{:,.3f}'.format)
    print('\n')
    print(importance[['Varname','Importance']].head(20))
    

    
    if draw_Tree:
        #draw decision tree
        dot_data = StringIO()  
        export_graphviz(dtree, out_file=dot_data,feature_names=varname,filled=True,rounded=True)

        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        display(Image(graph.create_png()) ) 
        graph.write_jpg("test_tree.jpg")
    
    return dtree

def RFtree_(X,y,n_tree=10,depth = 5,n_split = 300,n_leaf = 30,n_var = 0.33,draw_Tree=False , droplist=[] ):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=32)
    
    
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


    dtree = RandomForestClassifier(n_estimators=n_tree,max_depth=depth,min_samples_split=n_split 
                                   , min_samples_leaf=n_leaf,max_features=n_var)
    
    dtree.fit(X_train.drop(droplist,axis=1),y_train)
    
    y_train_pred = dtree.predict(X_train)
    y_test_pred  = dtree.predict(X_test)
    y_tot_pred  = dtree.predict(X)
    
    
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
    
    #probability
    a , b = dtree.classes_
    p_bad_train=dtree.predict_proba(X_train)[:,1]
    p_bad_test=dtree.predict_proba(X_test)[:,1]
    p_bad_tot=dtree.predict_proba(X)[:,1]
   
    #Model Stats
    #AUC
    test_auc = roc_auc_score(y_test,p_bad_test)
    train_auc = roc_auc_score(y_train, p_bad_train)
    tot_auc = roc_auc_score(y, p_bad_tot)
    
    #KS,IV
    train_ks,train_iv,c,d=ks_iv_single(p_bad_train,y_train, wgt=None , n_bin=10, var_type='continuous', keepna=True)
    test_ks,test_iv,c1,d1=ks_iv_single(p_bad_test,y_test, wgt=None , n_bin=10, var_type='continuous', keepna=True)
    tot_ks,tot_iv,c2,d2=ks_iv_single(p_bad_tot,y, wgt=None , n_bin=10, var_type='continuous', keepna=True)
    
#     print("Train_KS = {:,.3f} \nTest_KS = {:,.3f} \nTot_KS={:,.3f} \nKS Diff between Train and Test:{:,.2%}".format(train_ks, test_ks,tot_ks, (train_ks - test_ks)/train_ks))
#     print("Train_IV = {:,.3f} \nTest_IV = {:,.3f} \nTot_IV={:,.3f} \nIV Diff between Train and Test:{:,.2%}".format(train_iv, test_iv,tot_iv, (train_iv - test_iv)/train_iv))
#     print("Train_auc = {:,.3f} \nTest_auc = {:,.3f} \nTot_auc={:,.3f} \nAUC Diff between Train and Test:{:,.2%}".format(train_auc, test_auc,tot_auc, (train_auc - test_auc)/train_auc))
    
    index=['Train','Test','Total','% Change']
    sum_model_stats=pd.DataFrame({'KS':[train_ks,test_ks,tot_ks,(train_ks-test_ks)/train_ks] ,
                                  'IV':[train_iv,test_iv,tot_iv,(train_iv-test_iv)/train_iv],
                                  'AUC':[train_auc,test_auc,tot_auc,(train_auc-test_auc)/train_auc]},index=index)
    
    
    sum_model_stats.loc['% Change']= sum_model_stats.loc['% Change'].apply("{:.2%}".format)
    print(sum_model_stats,'\n')
    print('\n')
    print("Score distribution for Building")
    display(d)
    
    print('\n')
    print("Score distribution for Testing")
    display(d1)
    
    #importance
    varname=X_train.columns.tolist()
    imp = dtree.feature_importances_

    # indices = np.argsort(imp)[::-1]
    # for f in range(3):
    # print( (f+1,varname[indices[f]], indices[f],"{:,.2f}".format(imp[indices[f]])))
    
    importance = pd.DataFrame({'Varname':varname,'Importance':imp}).sort_values(by='Importance',ascending = False)
    importance['Importance']=importance['Importance'].map('{:,.3f}'.format)
    print('\n')
    print("Feature ranking(Top 20):",'\n')
    print(importance[['Varname','Importance']].head(20))
    

    
    #draw trees
    
    if draw_Tree and n_tree<=10:
   
        for a in dtree.estimators_:
            dot_data = StringIO()  
            export_graphviz(a, out_file=dot_data,feature_names=varname,filled=True,rounded=True)

            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
            display(Image(graph.create_png()) ) 
    
    std = np.std([tree.feature_importances_ for tree in dtree.estimators_],axis=0)
    indices = np.argsort(imp)[::-1]

    # Plot the feature importances of the forest
#     plt.figure()
#     plt.title("Feature importances")
#     plt.bar(range(X.shape[1]), imp[indices], color="r", yerr=std[indices], align="center")
#     plt.xticks(range(X.shape[1]), indices)
#     plt.xlim([-1, X.shape[1]])
#     plt.show()
    
    return dtree