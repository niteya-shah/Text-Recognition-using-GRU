import warnings
warnings.simplefilter('ignore')

import tensorflow as tf   # noqa
import numpy as np   # noqa
from decode import decode   # noqa
from skimage.io import imread, imshow   # noqa
from read_dataset import convert_to_char   # noqa
import nltk   # noqa

num_features = 64
num_units = 75
num_classes = 55
num_units_dense = 128

train_input_val = tf.keras.layers.Input(
    name='the_input', shape=[None, num_features], dtype='float32')

gru_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
    num_units, return_sequences=True,
    kernel_initializer='he_normal', name='gru1'))(train_input_val)
gru_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
    num_units, return_sequences=True, kernel_initializer='he_normal',
    name='gru2'))(gru_1)
gru_3 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
    num_units, return_sequences=True, kernel_initializer='he_normal',
    name='gru3'))(gru_2)

inner_1 = tf.keras.layers.Dense(
    num_units_dense, kernel_initializer='he_normal', name='dense2')(gru_3)
inner_2 = tf.keras.layers.Dense(
    num_classes, kernel_initializer='he_normal', name='dense3')(inner_1)
y_pred = tf.keras.layers.Activation('softmax', name='softmax')(inner_2)
model2 = tf.keras.Model(inputs=[train_input_val], outputs=y_pred)

# model2.compile(loss={'softmax': lambda y_true, y_pred: y_pred},
#                optimizer='adam')
model2.load_weights("./save2.h5")
model2.build(input_shape=[None, num_features])
train_input = np.load("train_data.npy", allow_pickle=True)
train_labels = np.load("labels.npy", allow_pickle=True)
