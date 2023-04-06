import numpy as np
import pandas as pd
import math

from comet_ml import Experiment

import pathlib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold

import tensorflow as tf
from tensorflow import keras
import random

import matplotlib.pyplot as plt
import seaborn as sns

experiment = Experiment(
    api_key="LDTPm7HDAGLUZ8N17Hu0w12Cg",
    project_name="icp-prediction",
    workspace="pefoss",
)

sns.set_style('whitegrid')

lr = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=1e-3,
      decay_steps=100000,
      decay_rate=0.98)
batch_size = 128
epochs = 200

hidden_size = 1024
input_size_encoder = 2
input_size_decoder = 1
output_size = 1
num_layers = 1
dropout_rate = 0.5
tf.random.set_seed(1234)
np.random.seed(1234)


BASE_PATH = pathlib.Path().resolve()
ICP_DATA_PATH = BASE_PATH / "Project1-Data/nICP"

df = pd.DataFrame()

for path in ICP_DATA_PATH.glob('*.csv'):
    data = pd.read_csv(path)
    df = pd.concat([df, data])

df.drop(['DateTime'], axis=1, inplace=True)
df.columns = ['MAP', 'ICP', 'nICP']
df.dropna(inplace=True)
df = df[::3]
"""##Preprocessing function"""

def process_data(sequence_length):
  x = df.drop(columns=['ICP'], axis=1)

  x = x[:int(x.values.shape[0]/sequence_length)*sequence_length]

  x = x.values.reshape(int(x.values.shape[0]/sequence_length), sequence_length, 2)
  y = df['ICP']
  y = y[:int(y.values.shape[0]/sequence_length)*sequence_length]
  y = y.values.reshape(int(y.values.shape[0]/sequence_length), sequence_length)
  Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)
  ytrain = ytrain.reshape(ytrain.shape[0], ytrain.shape[1], 1)
  Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], 2)
  Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 2)
  return Xtrain, Xtest, ytrain, ytest

"""##Split Train and Test data"""

scaler = MinMaxScaler()
df = scaler.fit_transform(df)
df = pd.DataFrame(df, columns = [['MAP', 'ICP', 'nICP']])
Xtrain, Xtest, ytrain, ytest = process_data(100)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

kfold = KFold(n_splits=5, shuffle=True)
kfold_performances = []
curr_fold = 0
for train, test in kfold.split(Xtrain):
  Xtest_validate = Xtrain[test]
  ytest_validate = ytrain[test]


  Xtrain_validate = Xtrain[train]
  ytrain_validate = ytrain[train]


  y_target_train = ytrain_validate[:, :-1, :]
  y_target_train = np.concatenate((np.zeros([ytrain_validate.shape[0], 1, ytrain_validate.shape[2]]), y_target_train), axis=1)

  y_target_test = ytest_validate[:, :-1, :]
  y_target_test = np.concatenate((np.zeros([ytest_validate.shape[0], 1, ytest_validate.shape[2]]), y_target_test), axis=1)

  encoder_inputs = Input(shape=(None, input_size_encoder))
  encoder = LSTM(hidden_size, return_state=True, return_sequences=True)
  encoder_outputs, state_h, state_c = encoder(encoder_inputs)
  
  encoder_states = [state_h, state_c]

  decoder_inputs = Input(shape=(None, input_size_decoder))
  
  decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True)
  decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                      initial_state=encoder_states)
  decoder_dense = Dense(output_size, activation=None)
  decoder_outputs = decoder_dense(decoder_outputs)

  
  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

  model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
  model.fit([Xtrain_validate, y_target_train], ytrain_validate,
          batch_size=batch_size,
          epochs=epochs,
          verbose = 1)
  performance = model.evaluate([Xtest_validate, y_target_test], ytest_validate)[1]
  kfold_performances.append(performance)
  experiment.log_metric('accuracy_fold_%d' % curr_fold, performance)
  curr_fold += 1
experiment.log_metric("rmse_validation_average", np.average(np.array(kfold_performances)))

encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(hidden_size,))
decoder_state_input_c = Input(shape=(hidden_size,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def seq2seq(input_test):
  state = encoder_model.predict(input_test)

  target_seq = np.zeros((input_test.shape[0], 1, 1))

  target_len = input_test.shape[1]

  stop_condition = False 

  outputs = []

  seq_index = 0

  while not stop_condition:
    output_tokens, h, c = decoder_model.predict(
            [target_seq] + state)

    state = [h, c]

    outputs.append(output_tokens)

    target_seq = output_tokens

    seq_index += 1

    if seq_index == target_len:
      stop_condition = True

    if stop_condition:
      break

  return np.concatenate(outputs, axis=1)



outputs = seq2seq(Xtest)

outputs, ytest, Xtest = outputs.flatten(), ytest.flatten(), Xtest[:, :, 1].flatten()
ytest = pd.DataFrame(ytest)
predictX = pd.DataFrame(outputs)
Xtest = pd.DataFrame(Xtest)
df_predicts = pd.concat([ytest, predictX, Xtest], axis=1)
df_predicts.columns = ['Y test', 'Y Predicts', 'X test']
df_predicts.set_index('X test', inplace=True)

ax = sns.scatterplot(data=df_predicts[::100], x='Y Predicts', y='Y test')
ax.set(xlim=(0, 1))
experiment.log_figure(figure_name='y_predicts vs. y_test', figure=ax.figure)
plt.show()

plt.figure()
random_index = int(random.random() * (df_predicts.size - 100))
ax2 = sns.lineplot(data=df_predicts.iloc[random_index: random_index + 100])
ax2.set(ylim=(0, 1))
experiment.log_figure(figure_name='predicts_seq', figure=ax2.figure)
plt.show()

plt.figure()
random_index = int(random.random() * (df_predicts.size - 100))
ax3 = sns.lineplot(data=df_predicts.iloc[random_index: random_index + 100])
ax3.set(ylim=(0, 1))
experiment.log_figure(figure_name='predicts_seq_2', figure=ax3.figure)
plt.show()

plt.figure()
random_index = int(random.random() * (df_predicts.size - 100))
ax4 = sns.lineplot(data=df_predicts.iloc[random_index: random_index + 100])
ax4.set(ylim=(0, 1))
experiment.log_figure(figure_name='predicts_seq_3', figure=ax4.figure)
plt.show()