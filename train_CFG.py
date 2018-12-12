import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, RNN, GRU
from keras.optimizers import RMSprop, SGD, Adam, Nadam
from keras.callbacks import ModelCheckpoint

from data import getReadyData

import numpy as np

X_train, X_val, X_test, y_train, y_val, y_test = getReadyData()
a=2

def rnn_from_cfg(cfg):
  saved_model_path = './SMAC3out/models/' \
                    + str(cfg['cell_size']) + '_' \
                    + str(cfg['n_cell']) + '_' \
                    + str(cfg['dropout']) + '_' \
                    + cfg['activation'] + '_' \
                    + cfg['optimizer'] + '_' \
                    + str(cfg['optimizer_lr']) + '_' \
                    + str(cfg['learning_decay_rate']) + '.hdf5'

  model = Sequential()

  model.add(LSTM(cfg['cell_size'], input_shape=(20, 2)))
  for i in range(cfg['n_cell'] - 1):
      model.add(LSTM(cfg['cell_size']))

  model.add(Dropout(cfg['dropout']))

  model.add(Dense(2))
  model.add(Activation(cfg['activation']))

  if cfg['optimizer'] == 'adam':
    opt = Adam(lr=cfg['optimizer_lr'], decay = cfg['learning_decay_rate'])
  elif cfg['optimizer'] == 'sgd':
    opt = SGD(lr=cfg['optimizer_lr'], decay = cfg['learning_decay_rate'])
  elif cfg['optimizer'] == 'nadam':
    opt = Nadam(lr=cfg['optimizer_lr'], schedule_decay = cfg['learning_decay_rate'])
  elif cfg['optimizer'] == 'RMSprop':
    opt = RMSprop(lr=cfg['optimizer_lr'], decay = cfg['learning_decay_rate'])

  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
  model.summary()

  model.fit(X_train, y_train, batch_size=1024, epochs=cfg['epochs'],
      validation_data=[X_val, y_val],
      callbacks=[ModelCheckpoint(saved_model_path, monitor='val_loss',
        verbose=2, save_best_only=True)])

  return model.evaluate(X_test, y_test)[0]

def main():
  best_cfg = np.load("C:\\NNwork\\HFSF\\SMAC3out\\best.cfg")
  best_cfg['epochs'] = 200
  results = rnn_from_cfg(best_cfg)

if __name__ == '__main__':
  main()
