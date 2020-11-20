#loads data on the fly with batching
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
import keras
from sentence_transformers import SentenceTransformer
import pandas as pd
import pdb
from sklearn.preprocessing import LabelEncoder

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, y_data, dim=(768*2,), total_samples=0, batch_size=200, n_classes=3, shuffle=True):
        'Initialization'
        self.df = df
        self.bert_embedding_model= SentenceTransformer('bert-base-nli-mean-tokens')
        self.batch_size = batch_size
        self.labels = y_data
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.dim = dim
        self.total_samples = total_samples
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.total_samples / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X, y = self.__data_generation(index*self.batch_size)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = None

    def process_title(self, row):
        title_embedding = self.bert_embedding_model.encode([row['title1_en'], row['title2_en']])

        return keras.layers.Concatenate(axis=0)([title_embedding[0], title_embedding[1]])

    def __data_generation(self, start):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list(range(start, start+self.batch_size))):
            # Store sample
            X[i,] = self.process_title(self.df.iloc[ID]) #process
            # Store class
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

'''
Keras script
Now, we have to modify our Keras script accordingly so that it accepts the generator that we just created.

import numpy as np

from keras.models import Sequential
from my_classes import DataGenerator

# Parameters
params = {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = # IDs
labels = # Labels

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
[...] # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)



'''
from sklearn.model_selection import train_test_split
from models import SimpleClassifier

#    def __init__(self, df, y_data, dim=(768*2), batch_size=200, n_classes=3, shuffle=True):
df=pd.read_csv("https://raw.githubusercontent.com/awaris123/OSNA-Project2/main/option1-data/train.csv")
label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(df['label'].values)

X_train, X_test, y_train, y_test = train_test_split(df[['title1_en', 'title2_en']], targets, test_size=0.2)
df_train = pd.DataFrame()

training_generator = DataGenerator(X_train, y_train, total_samples=y_train.shape[0], batch_size=params['batch_size'])
validation_generator = DataGenerator(X_test, y_test, total_samples=y_test.shape[0], batch_size=params['batch_size'])

model = SimpleClassifier(input_shape=768*2)
params={'lr': 0.01, 'opt': 'adam', 'model': 'SimpleClassifier', 'loss': 'crossentropy', 'batch_size': 100}


lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=params['lr'],
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=params['lr'])

model.model.compile(optimizer=opt, loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])#, keras.metrics.Precision(), keras.metrics.Recall()])
model.model.fit_generator(generator=training_generator,
                          validation_data=validation_generator,
                          epochs=50,
                    use_multiprocessing=False,
                    verbose=1)
model.model.save("./test/bert_mlp_unbalanced")
