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
      initial_learning_rate=1e-4,
      decay_steps=100000,
      decay_rate=0.98)
batch_size = 128
epochs = 1

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
    break

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
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_features, hidden_size, dropout):
    super(Encoder, self).__init__()
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.rnn = tf.keras.layers.LSTM(hidden_size, input_shape=(None, num_features), return_sequences=True, return_state=True)

  def call(self, x):
    x = self.dropout(x)

    encoder_outputs, hidden_state, cell_state = self.rnn(x)

    return encoder_outputs, hidden_state, cell_state


class Attention(tf.keras.layers.Layer):
  def __init__(self):
    super(Attention, self).__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(key_dim=3, num_heads=1)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x, context):

    attn_output = self.mha(
        query=x,
        value=context,
        )

  
    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_features, hidden_size, dropout):
    super(Decoder, self).__init__()
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.rnn = tf.keras.layers.LSTM(hidden_size, input_shape=(None, num_features), return_sequences=True, return_state=True)
    self.dense = tf.keras.layers.Dense(1, activation='linear')
    self.attention = Attention()
  
  def call(self, x, encoder_output, hidden, cell):
    x = self.dropout(x)

    rnn_output, hidden, cell = self.rnn(x, initial_state=[hidden, cell])

    dense_input = self.attention(rnn_output, encoder_output)

    output = self.dense(dense_input)

    return output, hidden, cell


class Seq2Seq(tf.keras.Model):
  def __init__(self, num_features_encoder, num_features_decoder, hidden_size, dropout):
    super(Seq2Seq, self).__init__()    
    self.encoder = Encoder(num_features_encoder, hidden_size, dropout)
    self.decoder = Decoder(num_features_decoder, hidden_size, dropout)

  def call(self, inputs):
    original, target = inputs

    enc_outputs, hidden, cell = self.encoder(original)

    outputs, hidden, cell = self.decoder(target, enc_outputs, hidden, cell)

    return outputs
  
  def predict(self, input):
    enc_outputs, hidden, cell = self.encoder(input)

    state  = [hidden, cell]

    target_seq = np.zeros((input.shape[0], 1, 1))
    stop_condition = False 

    outputs = []

    target_len = input.shape[1]

    seq_index = 0

    while not stop_condition:
      output_tokens, h, c = self.decoder(target_seq, enc_outputs, state[0], state[1])

      state = [h, c]

      outputs.append(output_tokens)

      target_seq = output_tokens

      seq_index += 1

      if seq_index == target_len:
        stop_condition = True

      if stop_condition:
        break

    return np.concatenate(outputs, axis=1)


model = Seq2Seq(2, 1, 1024, 0.5)

model.compile(optimizer=tf.keras.optimizers.Adam(lr),
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()])

ytarget = ytrain[:, :-1, :]
ytarget = np.concatenate((np.zeros([ytrain.shape[0], 1, ytrain.shape[2]]), ytarget), axis=1)

model.fit([Xtrain, ytarget], ytrain, epochs=epochs, batch_size=batch_size)

outputs = model.predict(Xtest)
print("Testing error ", np.sum(outputs.reshape(outputs.shape[0], outputs.shape[1])-ytest)/(ytest.shape[0] * ytest.shape[1]))