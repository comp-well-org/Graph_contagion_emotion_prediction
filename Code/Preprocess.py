import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


## For random split sampling test train val set
from sklearn.model_selection import train_test_split

# To fill missing values
from sklearn.impute import KNNImputer

from datetime import date, timedelta,datetime
from tqdm import tqdm

### IMPORT SPEKTRAL CLASSES ###
from spektral_utilities import *
from spektral_gcn import GraphConv
#from spektral.transforms import Degree,gcn_filter

## Standardize data and performance metrics
from sklearn.preprocessing import StandardScaler



def process_Graphs(call,sms,scale):
    # Normalize graphs with global max values to a scale between [0,10]
    gg  = call
    gg2 = sms 
    mx  = []
    mx2 = []

    # Find max
    for i in range(0,call.shape[0]):
        g1 = gg.A_msg[i]
        mx.append(g1)
       
        
    for i in range(0,sms.shape[0]):
        g2 = gg2.A_msg[i]
        mx2.append(g2)
        
    call_max = np.array(mx).max()/scale
    sms_max  = np.array(mx2).max()/scale

    #Normalize all A's with global max values 
    norm_call = call.copy(deep=True)
    norm_sms  = sms.copy(deep=True)
   
    for i in range(0,call.shape[0]):
        norm_call.A_msg[i] = call.A_msg[i]/call_max
    
    for i in range(0,sms.shape[0]):
        norm_sms.A_msg[i]  = sms.A_msg[i]/sms_max

    return norm_call, norm_sms




def construct_block_data(daily_d,user_list,feature_index,label_index,timesteps):
# construct User X Timestep dataframe where each entry is a object of feature vector

    num_users   = user_list.shape[0]
    num_missing = 0 
    Data      = []
    Label     = []
    flag      = np.zeros(num_users)
    u         = 0
    user_enc_id = 0 # do encoding so pivot can be used with repeated users
   
    for usr in user_list:
        user_enc_id = user_enc_id+1
        usr_data = daily_d[daily_d.user_id==usr]
        
        if usr_data.shape[0]==0:
            num_missing = num_missing+1
        
        avg_data = daily_d.iloc[:,feature_index].mean(axis=0)#.to_numpy()
        avg_data = avg_data.to_numpy()
        avg_data = np.reshape(avg_data, (1, -1))
       # print(avg_data.shape)
        avg_label = daily_d.iloc[:,label_index].mean(axis=0)   #[0]#.to_numpy()
#         avg_label = avg_label.to_numpy()
#         avg_label = np.reshape(avg_label, (1, -1))
        #print(avg_label.shape)

        for t in timesteps:
            
            if usr_data[usr_data.timestamp==t].iloc[:,feature_index].shape[0]>0 :
                DF1 ={"user_id":user_enc_id,"timestamp":t,"features":usr_data[usr_data.timestamp==t].iloc[:,feature_index].to_numpy()} 
                Data.append(DF1)
                
        
                DF2 ={"user_id":user_enc_id,"timestamp":t,"features":usr_data[usr_data.timestamp==t].iloc[:,label_index].to_numpy()} 
                Label.append(DF2)
            
            else:
                flag[u]=flag[u]+1
                DF1 ={"user_id":user_enc_id,"timestamp":t,"features":avg_data} 
                Data.append(DF1)
                DF2 ={"user_id":user_enc_id,"timestamp":t,"features":avg_label} 
                Label.append(DF2)
         
    u     = u+1 
    D1    = pd.DataFrame(Data)
    D2    = D1.pivot_table(values='features', index='user_id', columns='timestamp', aggfunc='first')
    label = pd.DataFrame(Label)
    label = label.pivot_table(values='features', index='user_id', columns='timestamp', aggfunc='first')
    return D2, label,flag, num_missing




def flatten_data(all_sequence,sequence_length):
    aa=[[] for _ in range(sequence_length)]
    for j in range(all_sequence.shape[-1]):
        for i in range(all_sequence.shape[0]):
            tmp = all_sequence.values[i,j]
            aa[j].append( tmp)     
    aa = np.array(aa)
    dd=aa.reshape([sequence_length,-1])
    return dd

def get_timespan(df, today, days):    
    df = df[pd.date_range(today - timedelta(days=days), 
            periods=days, freq='D')] # day - n_days <= dates < day    
    return df

def create_features(df, today, sequence_length,A,num_users):
    flag = 1
    sequence       = get_timespan(df, today, sequence_length)
    all_sequence   = flatten_data(sequence,sequence_length)      
    group_store    = all_sequence.reshape((1, sequence_length,-1 ))
    store_corr     = A.reshape((1,num_users,num_users))
    store_features = np.array([np.array(xi) for xi in sequence.values[:,-1]]).reshape(1,num_users,-1) 
    return group_store, store_corr, store_features,flag
    

def create_label(df, today,num_users):
    y = df[today].values
    return y.reshape((-1, num_users))



def test_date_existence(dates,D2):
    flag = 1
    for date in dates:
        if (D2.columns == pd.Timestamp(date)).any():
            f=1
        else:
            flag = 0   
    return flag



def prepare_data_for_NN(processed_data,graph,sequence_length,train_date,valid_date,feature_index,label_index,num_users):
    X_seq, X_cor, X_feat, y, flag,usr_arr = [], [], [], [], [], []
    total_missing   = 0
    #graph           = norm_call
    aa = []
    for d in tqdm(pd.date_range(train_date+timedelta(days=sequence_length), valid_date)):
    
        if (processed_data.timestamp == pd.Timestamp(d)).any():
            date_mask = graph[graph.start_date<=d ] # day - n_days <= dates < day 
            masked_d  = date_mask[date_mask.end_date>=d]
        
            for i in masked_d.index:
            
                users              = masked_d.usr_list[i]
                
                
                dates              = [d - timedelta(days=x) for x in range(sequence_length,-1,-1)]
                dates_fill         = [d - timedelta(days=x) for x in range( max(3, sequence_length),-1,-1)]
                ff                 = processed_data.loc[(processed_data['timestamp']>=dates_fill[0]) & (processed_data['timestamp']<=dates_fill[-1])]
            
                D2,label,flag_, num_miss = construct_block_data(ff,users,feature_index,label_index,dates)
                total_missing = total_missing +num_miss
                date_exist   = test_date_existence(dates,D2) 
           
                if date_exist==1 and num_miss==0 :
                    seq_, corr_, feat_,flag2 = create_features(D2, d, sequence_length,masked_d.A_msg[i],users.shape[0])
                    aa.append(masked_d.A_msg[i].sum())
                    y_                 = create_label(label, d,users.shape[0])
                    X_seq.append(seq_), X_cor.append(corr_), X_feat.append(feat_), y.append(y_),flag.append(flag_),usr_arr.append(np.array(users).reshape(1,-1))
                   
    
    X_seq_a   = np.concatenate(X_seq, axis=0).astype('float16')
    X_cor_a   = np.concatenate(X_cor, axis=0).astype('float16')
    X_feat_a  = np.concatenate(X_feat, axis=0).astype('float16')
    y_a       = np.concatenate(y, axis=0).astype('float16')
    usr_arr_a = np.concatenate(usr_arr, axis=0)
    
    
    return X_seq_a,X_cor_a,X_feat_a,y_a,usr_arr_a
    
def split_scale(X_seq_a,X_cor_a,X_feat_a,y_a,TEST_size,TRAIN_size,num_users,usr_list):    
    ## Split data in test, train and val sets
    [tmp, test] = train_test_split(  range(0,X_seq_a.shape[0]),shuffle=True, test_size=TEST_size,train_size=TRAIN_size )
    [train,vl]  = train_test_split( tmp, test_size=0.1, random_state=11)


    X_train_seq  = X_seq_a[train,:,:]
    X_train_cor  = X_cor_a[train,:,:]
    X_train_feat = X_feat_a[train,:,:]
    y_train      = y_a[train,:]

    X_valid_seq  = X_seq_a[vl,:,:]
    X_valid_cor  = X_cor_a[vl,:,:]
    X_valid_feat = X_feat_a[vl,:,:]
    y_valid      = y_a[vl,:]

    X_test_seq  = X_seq_a[test,:,:]
    X_test_cor  = X_cor_a[test,:,:]
    X_test_feat = X_feat_a[test,:,:]
    y_test      = y_a[test,:]
    usr_test    = usr_list[test,:]
  

    ### SCALE SEQUENCES ###

    scaler_seq  = StandardScaler()
    scaler_y    = StandardScaler()
    scaler_feat = StandardScaler()
    seq_len     = X_test_seq.shape[2]
    
    X_train_seq = scaler_seq.fit_transform(X_train_seq.reshape(-1,seq_len)).reshape(X_train_seq.shape)
    X_valid_seq = scaler_seq.transform(X_valid_seq.reshape(-1,seq_len)).reshape(X_valid_seq.shape)
    X_test_seq  = scaler_seq.transform(X_test_seq.reshape(-1,seq_len)).reshape(X_test_seq.shape)

    y_train = scaler_y.fit_transform(y_train)
    y_valid = scaler_y.transform(y_valid)
    y_test = scaler_y.transform(y_test)

    X_train_feat = scaler_feat.fit_transform(X_train_feat.reshape(-1,num_users)).reshape(X_train_feat.shape)
    X_valid_feat = scaler_feat.transform(X_valid_feat.reshape(-1,num_users)).reshape(X_valid_feat.shape)
    X_test_feat = scaler_feat.transform(X_test_feat.reshape(-1,num_users)).reshape(X_test_feat.shape)
    
    ### OBTAIN LAPLACIANS FROM CORRELATIONS for GCN-LSTM ###
    X_train_lap = localpooling_filter(np.abs(X_train_cor))
    X_valid_lap = localpooling_filter( np.abs(X_valid_cor))
    X_test_lap = localpooling_filter( np.abs(X_test_cor))


    return X_train_feat, X_train_seq, X_train_lap,y_train, scaler_y, X_valid_feat, X_valid_seq,X_valid_lap, y_valid, X_test_feat,X_test_seq, X_test_lap, y_test, X_test_cor,tmp, usr_test