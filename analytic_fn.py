import pandas as pd
import numpy as np
from IPython.display import display, HTML


def datasum(datin,key):
    n,k=datin.shape
    print('Number of Observations:',n)
    print('Number of variables:',k )
    unikey = datin[key].nunique()
    display('Unique key #',unikey)
    varname = list(datin.columns)
    display('variable list:', varname)
    



def ckdata(datin,showtable ): 
    s1 = datin.describe(percentiles=[.01, .05, .25, .75, .95,.99 ] )
    n = datin.shape[0]
    
    s2=s1.T
    s2['Missing']=(n-s2['count'])/n
   
   
    
    
    #same mean (mean <> 0 )
   
    sv = s2[(s2['mean'].duplicated(keep=False)) & (s2['mean'] != 0) ]
    #sv = s2[s2['mean'].duplicated(keep=False)]
   
    
    
    
    sv_drop = s2[s2['mean'].duplicated(keep= 'first')].index.tolist()
    
    #all missing or 0
    am=s2[s2['mean'].isnull()]
    az=s2[s2['mean'] == 0]
    
    am_drop = am.index.tolist()
    
    
    az_drop = az.index.tolist()
    
      
    print('# of Variables with all values 0: ', len(az_drop) ) 
    print('# of Variables with same values:', len(sv_drop) )
    print('# of Variables with all missings:', len(am_drop) )  
        
    if showtable == True:
        display(s2[['count','Missing', 'mean', 'std', 'min', '1%', '5%', '25%', '50%', '75%', '95%', '99%', 'max'] ].style.format({'Missing':"{:.2%}"}))
        
    print('Same Mean:')
    display(sv.sort_values(by=['mean'])) 


    print('All Zero:')
    display(az)
    
    print('All Missing')
    display(am)
    
    return sv_drop,am_drop,az_drop
