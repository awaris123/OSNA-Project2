from models import SimpleClassifier
import keras
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.utils import resample
import pdb
def upsample(df, majority_count):
    print(majority_count)
    df = resample(df, replace=True, n_samples=majority_count, random_state=0)
    return df

def balance(df, label_encoder=None):
    max_id=df_train['label'].value_counts().idxmax()
    minority_labels=list(range(0, 3))
    minority_labels.remove(max_id)
    # balance via upsampling
    df_majority=df[df['label']==max_id]
    df_minority_agree=df[df['label']==minority_labels[0]]
    df_minority_disagree=df[df['label']==minority_labels[1]]

    df_minority_agree=upsample(df_minority_agree, df_majority.shape[0])
    df_minority_disagree=upsample(df_minority_disagree, df_majority.shape[0])

    #concat
    df=pd.concat([df_majority, df_minority_agree, df_minority_disagree])
    print(df['label'].value_counts())

    return df
np.random.seed(0)
params={'lr': 0.01, 'opt': 'adam', 'model': 'SimpleClassifier', 'loss': 'crossentropy', 'batch_size': 200}

simple_classifier = SimpleClassifier(input_shape=600, linear_layers=2, layer1_nodes=600)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=params['lr'],
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=params['lr'])

simple_classifier.model.compile(optimizer=opt, loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

def filter_nans(arr):
    arr=np.asarray(arr).astype(np.float64)
    return np.isnan(np.sum(arr))
path_to_df = "trained_word2vec_embeddings.pickle" #"train_df_w_embeddings.pickle"
# load data
with open(path_to_df, 'rb') as f:
    df = pickle.load(f)

    df['joint_embedding']=df['joint_embedding'].apply(lambda x: x.numpy()[0].tolist())
    df['filter']=df['joint_embedding'].apply(lambda x: filter_nans(x))
    df = df[df['filter']==False]

    #implement sampling
    print("before_upsampling", df['label'].value_counts())

joint_embedding_matrix = df['joint_embedding'].values

label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(df['label'].values)

# convert targets into 1-hot form??
# OneHotEncoder()
joint_embedding_matrix=np.array(joint_embedding_matrix.tolist()).astype(np.float64)

'''
# nan check
array_sum = np.sum(joint_embedding_matrix)
array_sum = np.sum(array_sum)
print(len([print(i, np.sum(item)) for i, item in enumerate(joint_embedding_matrix) if (np.isnan(np.sum(item)))]))
array_has_nan = np.isnan(array_sum)
print(array_has_nan)
'''


X_train, X_test, y_train, y_test = train_test_split(joint_embedding_matrix, targets, test_size=0.2)
df_train = pd.DataFrame()
df_train['embed']= X_train.tolist()
df_train['label']=y_train

df_train=balance(df_train)
X_train= np.array(df_train['embed'].values.tolist()).astype(np.float64)
y_train=df_train['label'].values
print(X_train.shape)

y_train = keras.utils.to_categorical(y_train, 3)
y_test = keras.utils.to_categorical(y_test, 3)
print(y_train.shape)

simple_classifier.model.summary()
simple_classifier.model.fit(X_train, y_train,
          batch_size=params['batch_size'],
          epochs=25,
          verbose=1, validation_data=(X_test, y_test))

simple_classifier.model.save("./test/simple_classifier_w_google_word2vec_only_train_balanced")

