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
tf.keras.backend.set_floatx('float64')

class Preprocessing:
  def __init__(self, path_to_word2vec, binary=True):
    self.word2vec_model = KeyedVectors.load_word2vec_format(path_to_word2vec, binary=binary)
    self.pool = GlobalAveragePooling1D(data_format="channels_first")

  # splits a sentence into word tokens with stopwords and punctations omitted and words in lowercase
  # "A dog runs" -> ["dog", "runs"]
  def filter_tokens(self, sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    token_list = tokenizer.tokenize(sentence)
    filtered_tokens=[]
    for token in token_list:
      token = token.lower()
      if token not in stopwords.words('english'):
        filtered_tokens.append(token)
    return filtered_tokens

  def get_sentence_embedding(self, sentence):
    tokens = self.filter_tokens(sentence)
    embedding = [self.word2vec_model.get_vector(token) for token in tokens if token in self.word2vec_model.vocab]
    embedding = np.array(embedding, np.float64).reshape(1,300, -1)
    embedding = self.pool(embedding)

    return embedding

  def preprocess_row(self, row):
    title1_embedding = self.get_sentence_embedding(row['title1_en'])
    title2_embedding = self.get_sentence_embedding(row['title2_en'])
    return Concatenate(axis=1)([title1_embedding, title2_embedding])

preprocessing = Preprocessing("./GoogleNews-vectors-negative300.bin.gz")
#preprocessing.get_sentence_embedding("Test test test.")

df=pd.read_csv("https://raw.githubusercontent.com/awaris123/OSNA-Project2/main/option1-data/train.csv")
df['joint_embedding'] = df.apply(preprocessing.preprocess_row, axis=1)
print(df.head())