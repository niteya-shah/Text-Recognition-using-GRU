## %%
import warnings
warnings.simplefilter('ignore')
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from read_dataset import convert_to_char

import numpy as np
import json
from model_utils import ctc_decode_func,ctc_loss_func,norm_func

labels = np.load("labels.npy", allow_pickle = True)
train_input = np.load("train_data.npy", allow_pickle = True)

seq_lens = np.array([[i.shape[0]] for i in train_input])
out_lens = np.array([[i.shape[0]] for i in labels])

num_features = 64
batch_size = 50
num_units = 75
num_units_dense = 128
num_classes = 56
num_epochs = 100
num_examples = train_input.shape[0]
num_batches_per_epoch = int(num_examples/batch_size)
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.config.experimental.set_memory_growth = True
tf.config.experimental.log_device_placement = True

input_dataset = tf.data.Dataset.from_generator(lambda: train_input, tf.float32).padded_batch(batch_size, padded_shapes=([None, num_features]))
input_sequence_len =  tf.data.Dataset.from_tensor_slices(seq_lens).batch(batch_size)
output_targets = tf.data.Dataset.from_generator(lambda: labels, tf.int32).padded_batch(batch_size, padded_shapes = ([None]) , padding_values = -1)
output_sequence_len  = tf.data.Dataset.from_tensor_slices(out_lens).batch(batch_size)
dataset = tf.data.Dataset.zip((input_dataset, output_targets, input_sequence_len, output_sequence_len))
dataset = dataset.shuffle(1000).repeat(100).prefetch(buffer_size=AUTOTUNE)

train_input_val = tf.keras.layers.Input(name='the_input', shape=[None,num_features], dtype='float32')
target_inputs = tf.keras.layers.Input(name='the_labels', shape=[None], dtype='float32')
seq_len = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')
out_len = tf.keras.layers.Input(name='label_length', shape=[1], dtype='int64')

gru_1 =  tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_units,activation = 'tanh',recurrent_activation = 'sigmoid',recurrent_dropout = 0,unroll=False,use_bias=True,reset_after=True, return_sequences=True,kernel_initializer='he_normal', name='gru1'))(train_input_val)
gru_2 =  tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_units,activation = 'tanh',recurrent_activation = 'sigmoid',recurrent_dropout = 0,unroll=False,use_bias=True,reset_after=True, return_sequences=True,kernel_initializer='he_normal', name='gru2'))(gru_1)
gru_3 =  tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_units,activation = 'tanh',recurrent_activation = 'sigmoid',recurrent_dropout = 0,unroll=False,use_bias=True,reset_after=True, return_sequences=True,kernel_initializer='he_normal', name='gru3'))(gru_2)

inner_1 = tf.keras.layers.Dense(num_units_dense, kernel_initializer='he_normal', name='dense2')(gru_3)
inner_2 = tf.keras.layers.Dense(num_classes, kernel_initializer='he_normal', name='dense3')(inner_1)
y_pred = tf.keras.layers.Activation('softmax', name='softmax')(inner_2)

loss_out = tf.keras.layers.Lambda(ctc_loss_func, output_shape=(1,), name='ctc')([y_pred, target_inputs, seq_len, out_len])
cp = tf.keras.callbacks.ModelCheckpoint('C:\\Users\\Niteya Shah\\Desktop\\CTC\\Model_info\\model_best.h5',save_best_only=True,verbose=1)
tb = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
iterator = iter(dataset)
model = tf.keras.Model(inputs=[train_input_val, target_inputs, seq_len, out_len],outputs=loss_out)
adam = tf.optimizers.Adam(0.0001)
model.load_weights("./save5.h5")
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

model.fit(next(iterator),tf.zeros([batch_size, batch_size, batch_size]) ,callbacks = [tb], epochs = 40, steps_per_epoch = 50)

#tf.keras.models.save_model(model, "./save3.h5")
model.save_weights("./save5.h5")
#tf.test.is_gpu_available( cuda_only=True, min_cuda_compute_capability=None)
tf.keras.utils.plot_model(model, "./images/data_flow.png")
