import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
import os
import numpy as np
import pandas as pd
import random
import time
from tensorflow.keras.layers import LSTM
from app import*
def shuf(a,b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

scale=False

def create_data(ticker,pred):
    Feature_columns = ['adjclose', 'volume', 'open', 'high', 'low']
    split = 0.2
    shuffle=True
    date_split = True

    '''
    ticker = stock symbol
    Feature_columns = columns we need for neural net
    predict = number of days of prediction we need
    split = 0.2 spliting 80% to train and 20% to test

    ----------------
    hyper-parameter
    ----------------
    n_steps = step size we need to use
    shuffle = shuffle the dataset
    scale = making the values of feature columns in range of between 0 and 1

    '''
    if isinstance(ticker, str):
        df = si.get_data(ticker)
    elif isinstance(ticker , pd.DataFrame):
        df = ticker
    else:
        raise TypeError("ticker can be a str or a `pd.DataFrame` only")

    '''
    we are copying following things in dictionary so it is easy for use later
    df = dataframe
    scaler


    '''
    data = {}
    data["df"] = df.copy()

    for col in Feature_columns:
        assert col in df.columns,f"'{col}' does not exist in the dataframe."

    if "date" not in df.columns:
        df["date"] = df.index

    '''
    ------------
    Scaling
    -------------
    '''
    # if scale:
    #     col_scaler = {}
    #     for col in Feature_columns:
    #         scaler = preprocessing.MinMaxScaler()
    #         df[col] = scaler.fit_transform(np.expand_dims(df[col].values,axis=1))
    #         col_scaler[col] = scaler

    #     data["col_scaler"] = col_scaler


    '''
    ------
    Future target label and droping nan values
    ------
    '''
    df['future'] = df['adjclose'].shift(-pred)
    last_sequence = np.array(df[Feature_columns].tail(pred))

    df.dropna(inplace = True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry,target in zip(df[Feature_columns+["date"]].values,df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences),target])

    last_sequence = list([a[:len(Feature_columns)] for a in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    data['last_sequence'] = last_sequence

    '''
    creating train and test set
    '''
    X , y = [], []
    for seq , target in sequence_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    if date_split:
        '''
        spliting by date
        '''
        train_samples = int((1 - split) * len(X))
        data["X_train"] = X[:train_samples]
        data["y_train"] = y[:train_samples]
        data["X_test"]  = X[train_samples:]
        data["y_test"]  = y[train_samples:]

        if shuffle:
            # shuffle the dataset
            shuf(data["X_train"], data["y_train"])
            shuf(data["X_test"], data["y_test"])

    # else:
    #     '''
    #     spliting randomly
    #     '''
    #     data["X_train"], data["X_test"], data["Y_train"], data["Y_test"] = train_test_split(X, y,test_size=0.2, shuffle=True)


    dates = data["X_test"][:,-1,-1]
    data["test_df"] = data["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    data["test_df"] = data["test_df"][~data["test_df"].index.duplicated(keep='first')]
    data["X_train"] = data["X_train"][:, :, :len(Feature_columns)].astype(np.float32)
    data["X_test"] = data["X_test"][:, :, :len(Feature_columns)].astype(np.float32)



    return data


def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


EPOCHS = 2

n_steps=30
N_LAYERS = 2
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = True
### training parameters

LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 128
model_name='stock'






if not os.path.isdir("results"):
    os.mkdir("results")




# load the data
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_best_only=True, verbose=1)

model = create_model(n_steps, 5, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

def train(data):
    model.fit(data["X_train"], data["y_train"],batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(data["X_test"], data["y_test"]),verbose=2,callbacks=[checkpointer])




def predict(data):
    # retrieve the last sequence from data
    model1 = load_model('results/stock.h5')

    last_sequence = data["last_sequence"][-n_steps:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model1.predict(last_sequence)
    # get the price (by inverting the scaling)
    if scale:
        predicted_price = data["col_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return round(predicted_price,2)


def loss(data):
    # evaluate the model
    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # calculate the mean absolute error (inverse scaling)
    if scale:
        mean_absolute_error = data["col_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae

    return loss




