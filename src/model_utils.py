import numpy as np
import tensorflow as tf


def ctc_loss_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    input_length = input_length - 2
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, 
                                           input_length, label_length)


def ctc_decode_func(args):
    y_pred, seq_len = args
    y_pred, _ = tf.keras.backend.ctc_decode(y_pred, tf.squeeze(seq_len))
    return y_pred[0]


def norm_func(i):
    return np.subtract(np.array(i), np.mean(np.array(i)))/np.var(np.array(i))
