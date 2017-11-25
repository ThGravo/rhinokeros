from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


def build_regression_model(input_dim, output_dim,
                           dim_multipliers=(12, 3),
                           activations=('relu', 'relu'),
                           lr=.001):
    model = Sequential()
    model.add(Dense(input_dim * dim_multipliers[0], input_dim=input_dim, activation=activations[0]))
    for i in range(len(dim_multipliers) - 1):
        model.add(Dense(input_dim * dim_multipliers[i + 1],
                        activation=activations[min(i + 1, len(activations) - 1)]))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(loss='mae', optimizer=Adam(lr=lr), metrics=['accuracy'])
    return model