import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class CleanData():
  def __init__(self, directory):
    self.filename = directory
    self.feature_scale = MinMaxScaler()
  
  def read(self):
    self.dataframe = pd.DataFrame()
    for path in self.filename.glob('*.csv'):
        data = pd.read_csv(path)
        self.dataframe = pd.concat([self.dataframe, data])

    self.dataframe.drop(['DateTime'], axis=1, inplace=True)
    self.dataframe.columns = ['MAP', 'ICP', 'nICP']
    self.dataframe.dropna(inplace=True)
    self.dataframe = self.dataframe[::3]

  def train_test_split(self, seq_length, overlap, test_size, shuffle, random_seed): 
    self.scaled_df= self.feature_scale.fit_transform(self.dataframe)
    self.scaled_df = pd.DataFrame(self.scaled_df, columns = ['MAP', 'ICP', 'nICP'])
    self.Xtrain, self.Xtest, self.ytrain, self.ytest = self.process_data(self.dataframe, 
                                                                    seq_length, 
                                                                    overlap, 
                                                                    test_size, 
                                                                    shuffle, 
                                                                    random_seed)

    self.scaled_Xtrain, self.scaled_Xtest, self.scaled_ytrain, self.scaled_ytest = self.process_data(self.scaled_df, 
                                                                                                seq_length, 
                                                                                                overlap, 
                                                                                                test_size, 
                                                                                                shuffle, 
                                                                                                random_seed)
    return self.Xtrain, self.Xtest, self.ytrain, self.ytest
    
  def process_data(self, df, seq_length, overlap, test_size, shuffle, random_seed):
    x = df[['MAP', 'nICP']]
    x = x[:int(x.values.shape[0]/seq_length) * seq_length]
    x = x[:x.values.shape[0] - (seq_length - overlap)]
    x = x[:int(x.values.shape[0]/seq_length) * seq_length]
    x = x.values.reshape(int(x.values.shape[0]/seq_length), seq_length, 2)

    y = df['ICP']
    y = y[:int(y.values.shape[0]/seq_length) * seq_length]
    y = y[(seq_length - overlap):]
    y = y[:int(y.values.shape[0]/seq_length) * seq_length]
    y = y.values.reshape(int(y.values.shape[0]/seq_length), seq_length, 1)
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, shuffle=shuffle, test_size=test_size, random_state=random_seed)

    ytrain = ytrain.reshape(ytrain.shape[0], ytrain.shape[1], 1)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], 2)
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 2)
    ytest = ytest.reshape(ytest.shape[0], ytest.shape[1], 1)
    return Xtrain, Xtest, ytrain, ytest

