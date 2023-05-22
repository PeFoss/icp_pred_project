import pathlib
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np
import seaborn as sns

from ray import tune

from clean_data import CleanData

sns.set_style('whitegrid')


BASE_PATH = pathlib.Path().resolve()
ICP_DATA_PATH = BASE_PATH / 'Project1-Data/nICP'
clean = CleanData(ICP_DATA_PATH)
clean.read()

Xtrain, Xtest, ytrain, ytest = clean.train_test_split(100, 100, 0.3, True, 42)

sns.set_style('whitegrid')

def objective(config):
  lr = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=config["lr"],
      decay_steps=config["dec_steps"],
      decay_rate=config["dec_rate"])
  batch_size = 128
  epochs = 10

  hidden_size = 1024
  input_size_encoder = 2
  input_size_decoder = 1
  dropout_rate = 0.0
  tf.random.set_seed(1234)
  np.random.seed(1234)
  sequence_length = config["seq_len"]
  overlap = int(config["seq_len"] * config["overlap"])

  Xtrain, _, ytrain, _ = clean.train_test_split(sequence_length, overlap, 0.3, True, 42)

  n_iterations = 10
  n_samples_train = 1050
  n_samples_test = 450

  arr_index_xtrain = np.arange(0, len(Xtrain))

  scores = []

  for i in range(n_iterations):
    train = resample(arr_index_xtrain, n_samples = n_samples_train)
    test = resample(arr_index_xtrain, n_samples = n_samples_test)

    x_train = Xtrain[train]
    y_train = ytrain[train]
    x_test = Xtrain[test]
    y_test = ytrain[test]

    model = Seq2Seq(input_size_encoder, input_size_decoder, hidden_size, dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()])

    y_target = y_train[:, :-1, :]
    y_target = np.concatenate((np.zeros([y_train.shape[0], 1, y_train.shape[2]]), y_target), axis=1)

    model.fit([x_train, y_target], y_train, epochs=epochs, batch_size=batch_size)

    outputs = model(x_test)

    error = mean_squared_error(np.reshape(y_test, (y_test.shape[0], y_test.shape[1])), np.reshape(outputs, (outputs.shape[0], outputs.shape[1])), squared=False)
    scores.append(error)

  return np.mean(scores)

search_space = {
    "lr": tune.loguniform(1e-5, 1e-3), 
    "dec_steps": tune.uniform(10000, 100000),
    "dec_rate": tune.loguniform(0.95, 0.99),
    "seq_len": tune.grid_search([100]), 
    "overlap": tune.grid_search([0, 0.25, 0.5, 0.75, 1]),
}

objective = tune.with_resources(objective, {'cpu': 6})

tuner = tune.Tuner(objective, tune_config=tune.TuneConfig(num_samples=10, mode='min'), param_space=search_space)

results = tuner.fit()

print(results.get_best_result(mode='min'))