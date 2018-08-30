################################
#MODELING AND ANALYTIC TOOLS

# python_analytic_functions
#Mike Min 12/04/2017

python: 3.6.1
pandas: 0.20.1
sklearn: 0.18.1


Module list

#########################
1.score_dist_fn.py
Desc: Distribution of Key mertics such as GMV,Claims,Net loss Amt,BPS in terms of score distribution or segmentations 

post analysis

score_dist(datin, bin_num, class_var, perf_var_col, tran_var_col, bps_var_col, other_col, showtb=True)
score_dist(b2c_score,10,'trans_cnt_30d',['perf_flag'],['trans_amt_30d'],['claim_INR_amt_30d', 'net_loss_amt_30d'],['suspended_flag'])   
	no return

bin_cut(x,bin_num)

	bin_cut(b2c_score.score,10)
	return bins

##########################
2.univ_fn.py
score distribution

var_split(ind_x,perf_y,bin_num,showtb = True)
ks,iv=univ.var_split(b2c_score['score'],b2c_score['perf_flag'],bin_num= 10)
return ks,iv

univall(X,perf,bin_num,excelname,showtb=False)
ksiv=univ_all(b2c_ks,b2c_ks.perf_flag,20,excelname='')
return a table with ks and iv for all variables

#############################

3.Dtree_fn.py
server version GB tree only without draw tree

GBtree_(X,y,n_tree=50,learning_rate=0.1, depth = 8,n_split =200,n_leaf = 30,n_var = 0.33,sub_samp=0.8, droplist=[])

return dtree,s_prob
dtree: tree
s_prob: dataframe with prob, building, validation flag
Sample code:


##################################
4.Dtree_all_fn.pu

dtree_(X,y,depth = 3,n_split = 2,n_leaf = 1,n_var = 0.33,draw_Tree=False , droplist=[] )

RFtree_(X,y,n_tree=10,depth = 5,n_split = 300,n_leaf = 30,n_var = 0.33,draw_Tree=False , droplist=[] )

GBtree_(X,y,n_tree=50,learning_rate=0.1, depth = 8,n_split =200,n_leaf = 30,n_var = 0.33,sub_samp=0.8, droplist=[])

sample code:



#####################################
5. analytic tools
datasum(datin,key)

ckdata(datin)
