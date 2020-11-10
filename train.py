from models import SimpleClassifier
import keras
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

np.random.seed(0)
params={'lr': 0.01, 'opt': 'adam', 'model': 'SimpleClassifier', 'loss': 'crossentropy', 'batch_size': 2000}

simple_classifier = SimpleClassifier()

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=params['lr'],
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=params['lr'])

simple_classifier.model.compile(optimizer=opt, loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

# load data
with open("train_df_w_embeddings.pickle", 'rb') as f:
    df = pickle.load(f)
    df['joint_embedding']=df['joint_embedding'].apply(lambda x: x.numpy()[0].tolist())

joint_embedding_matrix = df['joint_embedding'].values

label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(df['label'].values)

# convert targets into 1-hot form??
# OneHotEncoder()
joint_embedding_matrix=np.array(joint_embedding_matrix.tolist()).astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(joint_embedding_matrix, targets, test_size=0.2)
y_train = keras.utils.to_categorical(y_train, 3)
y_test = keras.utils.to_categorical(y_test, 3)

simple_classifier.model.summary()
simple_classifier.model.fit(X_train, y_train,
          batch_size=params['batch_size'],
          epochs=25,
          verbose=1, validation_data=(X_test, y_test))
simple_classifier.model.save("./test/simple_classifier")
