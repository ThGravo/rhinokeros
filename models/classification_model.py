from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


def build_binary_classification_model(input_dim,
                                      dim_multipliers=(12, 4),
                                      activations=('relu', 'relu'),
                                      lr=.001):
    model = Sequential()
    model.add(Dense(input_dim * dim_multipliers[0], input_dim=input_dim, activation=activations[0]))
    for i in range(len(dim_multipliers) - 1):
        model.add(Dense(input_dim * dim_multipliers[i + 1], activation=activations[i + 1]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    return model


def build_classification_model(input_dim, num_classes,
                               dim_multipliers=(12, 4),
                               activations=('relu', 'relu'),
                               lr=.001):
    model = Sequential()
    model.add(Dense(input_dim * dim_multipliers[0], input_dim=input_dim, activation=activations[0]))
    for i in range(len(dim_multipliers) - 1):
        model.add(Dense(input_dim * dim_multipliers[i + 1], activation=activations[i + 1]))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    return model
