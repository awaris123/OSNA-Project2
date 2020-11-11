import keras

class SimpleClassifier:
    def __init__(self, input_shape=600, num_classes=3, linear_layers=1):
        self.lin_layers = []
        self.lin_layers.append(keras.layers.Dense(units=1000,input_dim=input_shape, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')) #highest performing is this 600 neuron layer + output layer
        self.lin_layers.append(keras.layers.BatchNormalization())
        self.lin_layers.append(keras.layers.Dropout(0.5))
        self.lin_layers.append(keras.layers.Dense(units=200, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
        self.lin_layers.append(keras.layers.BatchNormalization())
        self.lin_layers.append(keras.layers.Dropout(0.5))
        self.lin_layers.append(keras.layers.Dense(units=50, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
        self.lin_layers.append(keras.layers.BatchNormalization())
        self.lin_layers.append(keras.layers.Dropout(0.5))

        self.classifier = keras.layers.Dense(units=3, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')

        self.model = keras.models.Sequential([*self.lin_layers, self.classifier])