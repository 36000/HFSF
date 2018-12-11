import keras
import numpy as np
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, RNN, GRU
from keras.optimizers import RMSprop, SGD, Adam, Nadam
from keras.callbacks import ModelCheckpoint

sys.path.append("../SMAC3") # wherever SMAC3 install is

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from train_CFG import rnn_from_cfg

def print_incumb(cfg):
    print('Best model saved in: ' + './SMAC3out/models' \
            + str(cfg['cell_size']) + '_' \
            + str(cfg['n_cell']) + '_' \
            + str(cfg['dropout']) + '_' \
            + cfg['activation'] + '_' \
            + cfg['optimizer'] + '_' \
            + str(cfg['optimizer_lr']) + '_' \
            + str(cfg['learning_decay_rate']) + '.hdf5')

def main():
  cs = ConfigurationSpace()

  cell_size = CategoricalHyperparameter("cell_size", [128, 256], default_value=128)
  n_cell = CategoricalHyperparameter("n_cell", [1, 2], default_value=1)
  dropout = CategoricalHyperparameter("dropout", [0.2, 0.5], default_value=0.2)

  activation = CategoricalHyperparameter("activation", ['relu', 'sigmoid', 'tanh'], default_value='sigmoid')
  optimizer = CategoricalHyperparameter("optimizer", ['adam', 'sgd', 'nadam', 'RMSprop'], default_value='adam')
  optimizer_lr = CategoricalHyperparameter("optimizer_lr", [.0001, .0003, .001, .003, .01], default_value=.0003)
  learning_decay_rate = UniformFloatHyperparameter("learning_decay_rate", 0, 0.9, default_value=.6)

  # nn_type = CategoricalHyperparameter("nn_type", ['RNN', 'LSTM', 'GRU'], default_value='RNN')

  epochs = CategoricalHyperparameter("epochs", [20], default_value=10)

  cs.add_hyperparameters([cell_size, n_cell, dropout,
          activation, optimizer, optimizer_lr, learning_decay_rate, epochs])

  scenario = Scenario({"run_obj": "quality", "runcount-limit": 128, "cs": cs, "deterministic": "true"})
  scenario.output_dir_for_this_run = "C:\\NNwork\\HFSF\\SMAC3out"
  scenario.output_dir = "C:\\NNwork\\HFSF\\SMAC3out"
  smac = SMAC(scenario=scenario, rng=np.random.RandomState(23), tae_runner=rnn_from_cfg)

  best_model = smac.optimize()
  print_incumb(best_cfg)
  np.save( "C:\\NNwork\\HFSF\\SMAC3out\\best.cfg", best_cfg)

if __name__ == '__main__':
  main()
