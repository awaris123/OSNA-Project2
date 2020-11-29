import keras
from transformers import TFXLNetModel, XLNetTokenizer
#https://www.kaggle.com/alvaroibrain/xlnet-huggingface-transformers
import tensorflow as tf

class XLNet:
    def __init__(self, input_shape=600, num_classes=3, linear_layers=1):
        self.lin_layers = []
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        tokenized_sentence = keras.Input(shape=(128,), name='word_inputs', dtype='int32')
        xlnet=TFXLNetModel.from_pretrained('xlnet-large-cased')
        xlnet_encodings=xlnet(tokenized_sentence)[0]
        last_hidden = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)
        last_hidden = keras.layers.Dropout(0.1)(last_hidden)
        self.classifier = keras.layers.Dense(units=3, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(last_hidden)

        self.model = keras.Model(inputs=[tokenized_sentence], outputs=[self.classifier])