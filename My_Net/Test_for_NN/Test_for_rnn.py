import tensorflow as tf
lstm_cell = tf.contrib.rnn.LSTMCell(
                256,
                state_is_tuple=False,
               )
print lstm_cell.zero_state(64,tf.float32)