import argparse
import pandas as pd
import numpy
import math
import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow import keras

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    look_back = 1

    #load dataset
    df = pd.read_csv('data/power_information_detail.csv', usecols=[1])

    #normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)

    # split into train and test sets
    X_train, X_test = train_test_split(df, test_size=0.33, random_state=42)
    X_train,Y_train = utils.create_dataset(X_train,look_back=1)
    X_test,Y_test = utils.create_dataset(X_test,look_back=1)

    # reshape input to be [samples, time steps, features]
    X_train = numpy.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = numpy.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # craete and train LSTM
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

    #train_predict = model.predict(X_train)
    #test_predict = model.predict(X_test)

    #train_predict = scaler.inverse_transform(train_predict)
    #Y_train = scaler.inverse_transform([Y_train])
    #test_predict = scaler.inverse_transform(test_predict)
    #Y_test = scaler.inverse_transform([Y_test])

    #train_score = math.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))
    #print('Train Score: %.2f RMSE' % (train_score))
    #test_score = math.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))
    #print('Test Score: %.2f RMSE' % (test_score))

    # save model
    #model.save('my_model.h5')

    # load model
    model = keras.models.load_model('my_model.h5')
    
    # load 2022 data
    df_2022 = pd.read_csv('data/本年度每日尖峰備轉容量率.csv', usecols=[1])
    df_2022 = df_2022.apply(lambda x: x*10,axis=1)

    #normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_2022 = scaler.fit_transform(df_2022)

    X_train2 = df_2022[72:87]
    X_train2 = numpy.reshape(X_train2, (15, 1, 1))

    predict = model.predict(X_train2)
    answer = scaler.inverse_transform(predict)

    answer = pd.DataFrame(answer)
    answer['date'] = range(20220399,20220414)
    answer['date'][0] = '20220330'
    answer['date'][1] = '20220331'
    answer['備轉容量(MW)'] = answer[0]
    answer = answer.drop(0,axis=1)

    answer.to_csv('submission.csv',index=False)
    

    


