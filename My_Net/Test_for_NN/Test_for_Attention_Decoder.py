import tensorflow as tf

from My_NN.My_Net.Attention_Decoder import Attention_Decoder

batch_size = 64
num_char_classes = 27

img = tf.placeholder(tf.float32,[64,16,16,728])
lstm_cell = tf.contrib.rnn.LSTMCell(
                256,
                state_is_tuple=True,)
zero_label = tf.zeros([batch_size, num_char_classes])
def get_input(prev,i):
    return zero_label
attention_decoder = Attention_Decoder(seq_length=18,
                                      rnn_cell=lstm_cell,
                                      loop_function=get_input,
                                      )

outputs,states = attention_decoder.ops(img)
for v in attention_decoder.train_variables:
    print v.name,v.shape
print attention_decoder.train_variables_number
print outputs[0].get_shape()