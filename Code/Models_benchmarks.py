
## Import libraries
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

import networkx as nx
### IMPORT SPEKTRAL CLASSES ###
from spektral_utilities import *
from spektral_gcn import GraphConv

## Hyperparameter tuning
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, Hyperband
import random



def GCNHyperModel():

    ##Input layers
    inp_seq  = Input((sequence_length, lstm_input_sz))
    inp_lap  = Input((num_user,num_user))
    inp_feat = Input((num_user, gcn_input_sz))
     
    ##GCN layers
    x = GraphConv(48, activation='relu')([inp_feat, inp_lap])
    x = GraphConv(32, activation='relu')([x, inp_lap])
    x = GraphConv(16, activation='relu')([x, inp_lap])
    x = Flatten()(x)
        
    ##LSTM
    xx = LSTM(64, activation='tanh',return_sequences=True,kernel_initializer='random_normal', bias_initializer='zeros',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),activity_regularizer=regularizers.l2(1e-5))(inp_seq)
    xx = Dropout(0.2)(xx)
    xx = LSTM(32, activation='tanh',kernel_initializer='random_normal', bias_initializer='zeros',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),activity_regularizer=regularizers.l2(1e-5))(xx)
        
    ##Dense and concatenation
    x   = Concatenate()([x,xx])
    x   = BatchNormalization()(x)
    x   = Dense(units=64, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),activity_regularizer=regularizers.l2(1e-5))(x)
    x   = Dropout(0.4)(x)
    x   = Dense(units=32, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),activity_regularizer=regularizers.l2(1e-5))(x)
    out = Dense(num_user)(x)
        
    ##Final model
    model = Model([inp_seq, inp_lap, inp_feat], out)
       
    ## Training 
    model.compile(optimizer=Adam(learning_rate=0.001),loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def BaseHyperModel():

    ## Input layers
    inp_seq = Input((sequence_length, lstm_input_sz))
    inp_feat = Input((num_user, gcn_input_sz))
     
    ##Conv layers
    x = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu',input_shape=(num_user, gcn_input_sz))(inp_feat)
    x = Flatten()(x)
        
    ##LSTM
    xx = LSTM(32, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),activity_regularizer=regularizers.l2(1e-5),kernel_initializer='ones', bias_initializer='zeros')(inp_seq)
        
     ##Dense and concatenation

    x = Concatenate()([x,xx])
    x = BatchNormalization()(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(num_user)(x)
        
    ##Final model
    model = Model([inp_seq, inp_feat], out)
     
    ## Training 
    model.compile(optimizer=Adam(learning_rate=0.001),loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def BaseLSTMModel():
    inp_seq = Input((sequence_length, lstm_input_sz))

    ##LSTM
    xx = LSTM(64, activation='tanh',return_sequences=True,kernel_initializer='ones', bias_initializer='ones')(inp_seq)
    xx = Dropout(0.2)(xx) 
    xx = LSTM(32, activation='tanh',return_sequences=False, kernel_initializer='ones',bias_initializer='ones',recurrent_activation="relu")(xx)
    out = Dense(num_user)(xx)
        
    ##Final model
    model = Model([inp_seq], out)
     
    ## Training hyperparam
    model.compile(optimizer=Adam(learning_rate=0.01),loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model




        
def get_prediction_results(BatchSize,X_train_seq, X_train_lap, X_train_feat, y_train,X_valid_seq, X_valid_lap, X_valid_feat, y_valid,X_test_seq, X_test_lap, X_test_feat,scaler_y,num_users):
    
    global lstm_input_sz
    global gcn_input_sz
    global sequence_length
    global num_user
    
    num_user        = X_train_feat.shape[1]
    lstm_input_sz   = X_train_seq.shape[2]
    gcn_input_sz    = X_train_feat.shape[2]
    sequence_length = X_train_seq.shape[1]
    
   
    random.seed(11)
    
    es2           = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,min_delta=0.001)
    model_cnv     = BaseHyperModel()
    history_cnv   = model_cnv.fit([X_train_seq,  X_train_feat], y_train, epochs=200, batch_size=BatchSize, 
              validation_data=([X_valid_seq,  X_valid_feat], y_valid), callbacks=[es2], verbose=0)
    pred_test_all1c = model_cnv([X_test_seq,  X_test_feat])
    pred_cnv        = scaler_y.inverse_transform(pred_test_all1c)
    es1             = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,min_delta=0.001)
    model_gcn       = GCNHyperModel()
    history         = model_gcn.fit([X_train_seq, X_train_lap, X_train_feat], y_train, epochs=200, batch_size=BatchSize, 
              validation_data=([X_valid_seq, X_valid_lap, X_valid_feat], y_valid), callbacks=[es1], verbose=0)
    pred_test_all1  = model_gcn([X_test_seq, X_test_lap, X_test_feat])
    pred_gcn        = scaler_y.inverse_transform(pred_test_all1)
    es3             = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,min_delta=0.01)
    model_lstm      = BaseLSTMModel()
    history_lstm    = model_lstm.fit([X_train_seq], y_train, epochs=200, batch_size=BatchSize, 
              validation_data=([X_valid_seq], y_valid), callbacks=[es3], verbose=0)
    pred_test_all1l = model_lstm([X_test_seq])
    pred_lstm       = scaler_y.inverse_transform(pred_test_all1l)
    return pred_gcn,pred_cnv,pred_lstm, history,history_cnv,history_lstm
    
                    

        
        
        
