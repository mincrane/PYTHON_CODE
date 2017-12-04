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
    



def ckdata(datin): 
    s1 = datin.describe(percentiles=[.01, .05, .25, .75, .95,.99 ] )
    n = datin.shape[0]
    
    s2=s1.T
    s2['Missing']=(n-s2['count'])/n
    
    
    #same mean
    print('Same Mean:')
    sv = s2[s2['mean'].duplicated(keep=False)]
    display(sv.sort_values(by=['mean']))
    
    
    sv_drop = s2[s2['mean'].duplicated(keep= 'first')].index.tolist()
    
    #all missing or 0
    am=s2[s2['mean'].isnull()]
    az=s2[s2['mean'] == 0]
    print('All Missing')
    display(am)
    am_drop = am.index.tolist()
    
    print('All Zero:')
    display(az)
    az_drop = az.index.tolist()
    
    #display(s2.style.format({'Missing':"{:.2%}"}))
   
    return sv_drop,am_drop,az_drop
