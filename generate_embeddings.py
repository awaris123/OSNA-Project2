import pandas as pd
import numpy as np
import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from gensim.models import Word2Vec, KeyedVectors
from keras.layers import GlobalAveragePooling1D, Concatenate
import tensorflow as tf
import pdb
tf.keras.backend.set_floatx('float64')

class Word2Vec_Train:
  def __init__(self, df, size=100):
    self.df = df
    self.size = size

  def train_save(self, path_to="word2vec.wordvectors"):
    sentence_list = self.df.apply(lambda x: Preprocessing.filter_tokens(x['title1_en']), axis=1).values.tolist()
    sentence_list.extend(self.df.apply(lambda x: Preprocessing.filter_tokens(x['title2_en']), axis=1).values.tolist())

    model = Word2Vec(sentences=sentence_list, size=self.size, workers=4)
    model.wv.save(path_to)
    return model


class Preprocessing:
  def __init__(self, path_to_word2vec, binary=True, size=300):
    if binary:
      self.word2vec_model = KeyedVectors.load_word2vec_format(path_to_word2vec, binary=binary)
    else:
      self.word2vec_model = KeyedVectors.load(path_to_word2vec)
    self.pool = GlobalAveragePooling1D(data_format="channels_first")
    self.size = size

  # splits a sentence into word tokens with stopwords and punctations omitted and words in lowercase
  # "A dog runs" -> ["dog", "runs"]
  def filter_tokens(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    token_list = tokenizer.tokenize(sentence)
    filtered_tokens=[]
    for token in token_list:
      token = token.lower()
      if token not in stopwords.words('english'):
        filtered_tokens.append(token)
    return filtered_tokens

  def get_sentence_embedding(self, sentence):
    tokens = Preprocessing.filter_tokens(sentence)
    embedding = [self.word2vec_model.get_vector(token) for token in tokens if token in self.word2vec_model.vocab]
    embedding = np.array(embedding, np.float64).reshape(1,self.size, -1)
    embedding = self.pool(embedding)

    return embedding

  def preprocess_row(self, row):
    title1_embedding = self.get_sentence_embedding(row['title1_en'])
    title2_embedding = self.get_sentence_embedding(row['title2_en'])
    return Concatenate(axis=1)([title1_embedding, title2_embedding])

'''
preprocessing = Preprocessing("./GoogleNews-vectors-negative300.bin.gz")
#preprocessing.get_sentence_embedding("Test test test.")

df=pd.read_csv("https://raw.githubusercontent.com/awaris123/OSNA-Project2/main/option1-data/train.csv")
df['joint_embedding'] = df.apply(preprocessing.preprocess_row, axis=1)
print(df.head())

import pickle
with open('train_df_w_embeddings.pickle', 'wb') as f:
  pickle.dump(df, f)
'''
'''
#generate our embeddings
preprocessing = Preprocessing("./word2vec300.wordvectors", binary=False, size=300)

df=pd.read_csv("https://raw.githubusercontent.com/awaris123/OSNA-Project2/main/option1-data/train.csv")
df['joint_embedding'] = df.apply(preprocessing.preprocess_row, axis=1)
print(df.head())

import pickle
with open('trained_word2vec_embeddings.pickle', 'wb') as f:
  pickle.dump(df, f)

'''
'''
df=pd.read_csv("https://raw.githubusercontent.com/awaris123/OSNA-Project2/main/option1-data/train.csv")

#train word2vec_model
word2vec_model=Word2Vec_Train(df, size=300)
word2vec_model.train_save(path_to="word2vec300.wordvectors")
'''

def test_data_convert(path, binary=True, size=100):
  df =  pd.read_csv(path)
  if binary:
    preprocessing = Preprocessing("./GoogleNews-vectors-negative300.bin.gz")
  else:
    #preprocessing = Preprocessing("./word2vec.wordvectors", binary=False, size=size)
    preprocessing = Preprocessing("./word2vec300.wordvectors", binary=False, size=300)

  df['joint_embedding'] = df.apply(preprocessing.preprocess_row, axis=1)

  return df

#print(test_data_convert("./option1-data/test.csv"))