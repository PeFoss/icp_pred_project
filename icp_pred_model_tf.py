import numpy as np
import tensorflow as tf


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
    self.mha = tf.keras.layers.MultiHeadAttention(key_dim=3, num_heads=4)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x, context):

    attn_output = self.mha(
        query=x,
        value=context,
        )
    return attn_output

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

    att_output = self.attention(rnn_output, encoder_output)
    dense_input = tf.concat([rnn_output, att_output], axis = -1)
    output = self.dense(dense_input)

    return output, hidden, cell


class Seq2Seq(tf.keras.Model):
  def __init__(self, num_features_encoder, num_features_decoder, hidden_size, dropout):
    super(Seq2Seq, self).__init__()
    self.encoder = Encoder(num_features_encoder, hidden_size, dropout)
    self.decoder = Decoder(num_features_decoder, hidden_size, dropout)

  def call(self, inputs, training=False):
    if training:
      original, target = inputs

      enc_outputs, hidden, cell = self.encoder(original)

      outputs, hidden, cell = self.decoder(target, enc_outputs, hidden, cell)

      return outputs

    enc_outputs, hidden, cell = self.encoder(inputs)
    target_seq = np.zeros((inputs.shape[0], 1, 1))

    outputs = []

    target_len = inputs.shape[1]

    seq_index = 0

    while True:
      output_tokens, h, c = self.decoder(target_seq, enc_outputs, hidden, cell)

      hidden = h
      cell = c
      outputs.append(output_tokens)
      target_seq = output_tokens

      seq_index += 1
      if seq_index == target_len:
        outputs = tf.concat(outputs, axis=1)
        break

    return outputs

