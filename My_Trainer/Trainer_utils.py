import tensorflow as tf
from tensorflow.python.platform import flags
Flags = flags.FLAGS

def single_accuracy(net_predict,net_lable):
    with tf.variable_scope("get_single_accuracy"):
        single_acc = tf.cast(tf.equal(tf.argmax(net_predict, -1), tf.argmax(net_lable, -1)), tf.float32)
        single_acc = tf.reduce_mean(single_acc)
        return single_acc

def seq_accuracy(net_predict,net_lable):
    with tf.variable_scope("get_seq_accuracy"):
        seq_acc = tf.cast(tf.equal(tf.argmax(net_predict, -1), tf.argmax(net_lable, -1)), tf.float32)
        seq_acc = tf.reduce_sum(seq_acc, -1)
        seq_acc = tf.cast(tf.equal(seq_acc, Flags.label_length), tf.float32)
        seq_acc = tf.reduce_mean(seq_acc)
        return seq_acc