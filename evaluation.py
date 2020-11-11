# this file is to breakdown our results and analyze areas for improvement
from models import SimpleClassifier
import keras
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

np.random.seed(0)

def filter_nans(arr):
    arr = np.asarray(arr).astype(np.float64)
    return np.isnan(np.sum(arr))

def load_test_data():

    # load data
    with open("train_df_w_embeddings.pickle", 'rb') as f:
        df = pickle.load(f)
        df['joint_embedding'] = df['joint_embedding'].apply(lambda x: x.numpy()[0].tolist())
        df['filter'] = df['joint_embedding'].apply(lambda x: filter_nans(x))
        df = df[df['filter'] == False]

    joint_embedding_matrix = df['joint_embedding'].values

    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(df['label'].values)

    # convert targets into 1-hot form??
    # OneHotEncoder()
    joint_embedding_matrix = np.array(joint_embedding_matrix.tolist()).astype(np.float64)

    '''
    # nan check
    array_sum = np.sum(joint_embedding_matrix)
    array_sum = np.sum(array_sum)
    print(len([print(i, np.sum(item)) for i, item in enumerate(joint_embedding_matrix) if (np.isnan(np.sum(item)))]))
    array_has_nan = np.isnan(array_sum)
    print(array_has_nan)
    '''

    X_train, X_test, y_train, y_test = train_test_split(joint_embedding_matrix, targets, test_size=0.2)
    y_train = keras.utils.to_categorical(y_train, 3)
    #y_test = keras.utils.to_categorical(y_test, 3)
    return X_test, y_test


model = keras.models.load_model("./test/simple_classifier")
X_test, y_test = load_test_data()

predictions = model.predict(X_test)

df=pd.DataFrame()
df['predictions'] = predictions.tolist()
df['true'] = y_test

df['results'] = df.apply(lambda x: np.argmax(np.array(x['predictions'])) == x['true'], axis=1)#np.argmax(np.array(x['predictions'])) == x['true'])
# y_test = keras.utils.to_categorical(y_test, 3)

print(df.groupby(['true', 'results']).count())#.count()[['results']])
