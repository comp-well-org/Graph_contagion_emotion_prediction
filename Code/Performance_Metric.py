# import performance metrics

from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix, ConfusionMatrixDisplay,precision_score, recall_score
import networkx as nx
import numpy as np
import pandas as pd



def discretize_result(plt_d, low_thres,high_thres):
    
    bins = np.array([-100000, low_thres,high_thres,2000000 ]) 
    k    = 3
    if (low_thres==high_thres):
        bins = np.array([-10000000, high_thres,2000000 ]) 
        k = 2
    pl = plt_d.copy()
    pl['yt'] = pd.cut(plt_d.yt, bins,labels=np.arange(k), right=True)
    pl['yp'] = pd.cut(plt_d.yp, bins,labels=np.arange(k), right=True)
    return pl
    
    


def compute_error_vs_graph_metric_dis(true,pred,A_data,metric,low_thres,high_thres,which_f1,y_met,usr_list):
    # Error evaluation for y=discrete [0,1,2]
    instance, num_usrs = true.shape
    data_x             = []
    data_yt            = []
    data_yp            = []
    
    for i in range(0,instance):
#        G  = nx.from_numpy_matrix(A_data[i],create_using =nx.DiGraph)
        G  = nx.from_numpy_matrix(A_data[i])
        
        if (metric == 'degree'):
            x_val = np.array([val for (node, val) in G.degree()])
            
        if (metric == 'in-degree'):
            x_val = np.array(list(nx.algorithms.centrality.in_degree_centrality(G).values()))
            
        if (metric == 'out-degree'):
            x_val = np.array(list(nx.algorithms.centrality.out_degree_centrality(G).values()))
        
        if (metric == 'eigen'):
            x_val = np.array(list(nx.eigenvector_centrality(G).values()))
       
                             
        if (metric == 'comm'):
            Gu    = nx.from_numpy_matrix(A_data[i])
            x_val = np.array(list(nx.communicability_betweenness_centrality(Gu).values()))
                             
        if (metric == 'between'):
            x_val = np.array(list(nx.algorithms.centrality.betweenness_centrality(G).values())) 
         
        if (metric == 'close'):
            x_val = np.array(list(nx.algorithms.centrality.closeness_centrality(G).values()))
        
        if(metric == 'pagerank'):
            x_val = np.array( list( nx.pagerank(G, alpha=0.9).values()))
        
        if(metric=='usr'):
            x_val = usr_list[i,:]
        
        if(metric == 'none'):
            x_val = np.zeros(num_usrs)
        
            
        y_val = true[i,:]
        y_vap = pred[i,:]
        
        data_x.extend(x_val)
        data_yt.extend(y_val)
        data_yp.extend(y_vap)
        
    data        = {'x':data_x, 'yt': data_yt, 'yp':data_yp}
    plt_d       = pd.DataFrame(data) 
    pl          = discretize_result(plt_d, low_thres,high_thres)
    grp         = pl.groupby('x')
    
    xx = []
    f1 = []
    
    if (y_met=='f1'):
        for name,aa in grp:
            xx.extend([aa['x'].iloc[0]])
            f1.extend( [f1_score(aa['yt'] ,aa['yp'], average=which_f1)])
            
    if(y_met=='precision'):
        for name,aa in grp:
            xx.extend([aa['x'].iloc[0]])
            f1.extend( [precision_score(aa['yt'] ,aa['yp'], average=which_f1)])
        
    if(y_met=='recall'):
        for name,aa in grp:
            xx.extend([aa['x'].iloc[0]])
            f1.extend( [recall_score(aa['yt'] ,aa['yp'], average=which_f1)])
            
    if (y_met=='fn'):
        for name,aa in grp:
            xx.extend([aa['x'].iloc[0]])
            cm = confusion_matrix(aa['yt'] ,aa['yp'],labels=[0,1], normalize= 'all')
            f1.extend([cm[1,0]] )
            
    if (y_met=='fp'):
        for name,aa in grp:
            xx.extend([aa['x'].iloc[0]])
            cm = confusion_matrix(aa['yt'] ,aa['yp'],labels=[0,1], normalize= 'all')
            f1.extend([cm[0,1]] )
    
    out       = pd.DataFrame()
    out['x']  = xx
    out['f1'] = f1 
    

    return out


def get_confusion_matrix(reverse_test,pred_test_all,low_thres,high_thres):
    data        = {'yt': reverse_test.flatten(), 'yp': pred_test_all.flatten()}
    plt_d       = pd.DataFrame(data) 
    pl          = discretize_result(plt_d, low_thres,high_thres)
    cm = confusion_matrix(pl.yt, pl.yp, normalize= 'all')
   # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    return cm
    

def generate_multi_x_metric_mat(true,pred,A_data,usr_id,x_met):
    instance, num_usrs = true.shape
    data_x             = []
    data_yt            = []
    data_yp            = []
    col_name           = x_met
  #  y_col              = ['yp','yt']
   # col_name.extend(y_col)
    main_df            = pd.DataFrame(columns=col_name)
   
    
    for i in range(0,instance):
        DF                 = {}
        G  = nx.from_numpy_matrix(A_data[i])
        
        for metric in x_met:
            if (metric == 'degree'):
                DF[metric]  = np.array([val for (node, val) in G.degree()])
            
            if (metric == 'in-degree'):
                DF[metric]  = np.array(list(nx.algorithms.centrality.in_degree_centrality(G).values()))
            
            if (metric == 'out-degree'):
                DF[metric]  = np.array(list(nx.algorithms.centrality.out_degree_centrality(G).values()))
        
            if (metric == 'eigen'):
                try:
                    DF[metric]  = np.array(list(nx.eigenvector_centrality(G,max_iter=100).values()))
                except:
                     DF[metric] = -100
       
                             
            if (metric == 'comm'):
                Gu    = nx.from_numpy_matrix(A_data[i])
                DF[metric]  = np.array(list(nx.communicability_betweenness_centrality(Gu).values()))
                             
            if (metric == 'between'):
                DF[metric]  = np.array(list(nx.algorithms.centrality.betweenness_centrality(G).values())) 
         
            if (metric == 'close'):
                DF[metric]  = np.array(list(nx.algorithms.centrality.closeness_centrality(G).values()))
        
            if(metric == 'pagerank'):
                try:
                    DF[metric]  = np.array( list( nx.pagerank(G, alpha=0.9,max_iter=100).values()))
                except:
                    DF[metric] = -100
        
            if(metric == 'none'):
                DF[metric]  = np.zeros(num_usrs)
        
        DF['yt']  = true[i,:]
        DF['yp']  = pred[i,:]
        DF['usr'] = usr_id[i,:]
        
        tmp = pd.DataFrame(DF)
        main_df = main_df.append(tmp)
    
    return main_df
        
        
            
            
            
        
    