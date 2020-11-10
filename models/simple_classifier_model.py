import keras

class SimpleClassifier:
    def __init__(self, input_shape=600, num_classes=3, linear_layers=1):
        self.lin_layers = []
        self.lin_layers.append(keras.layers.Dense(units=600,input_dim=input_shape, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
        self.lin_layers.append(keras.layers.Dense(units=100, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))

        self.classifier = keras.layers.Dense(units=3, activation='softmax')
        self.model = keras.models.Sequential([*self.lin_layers, self.classifier])