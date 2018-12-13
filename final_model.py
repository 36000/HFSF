import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, SimpleRNN, GRU, Masking
from keras.optimizers import RMSprop, SGD, Adam, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from data import getReadyData

import numpy as np


def hand_model(cell_size, n_cell, epochs=10, dropout=0.5, activation='sigmoid', optimizer='adam', lr=0.03, decay=0.09, kernel_init='glorot_uniform'):
    x_train, x_val, x_test, y_train, y_val, y_test = getReadyData()
    saved_model_path = './hand_made_models/' \
                       + str(cell_size) + '_' \
                       + str(n_cell) + '_' \
                       + 'GRU' + '_' \
                       + str(dropout) + '_' \
                       + activation + '_' \
                       + optimizer + '_' \
                       + str(lr) + '_' \
                       + str(decay) + '.hdf5'

    model = Sequential()
    model.add(Masking(10.0, input_shape=(20, 2)))

    for i in range(n_cell - 1):
        model.add(GRU(cell_size, return_sequences=True, kernel_initializer=kernel_init))
    model.add(GRU(cell_size, kernel_initializer=kernel_init))

    model.add(Dropout(dropout))

    model.add(Dense(2))
    model.add(Activation(activation))

    model.compile(optimizer=Adam(lr=lr, decay=decay), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    model.fit(x_train, y_train, batch_size=1024, epochs=epochs,
              validation_data=[x_val, y_val],
              callbacks=[ModelCheckpoint(saved_model_path, monitor='val_loss',
                                         verbose=2, save_best_only=True),
                         EarlyStopping(monitor='val_loss', patience=5)])

    np.save('final_eval.npy', model.evaluate(x_test, y_test))


def main():
    hand_model(256, 2, epochs=200)


if __name__ == '__main__':
    main()
