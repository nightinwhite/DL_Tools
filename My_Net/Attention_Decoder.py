import tensorflow as tf
from tensorflow.python.util import nest
import utils
from tensorflow.contrib.layers.python.layers import regularizers

class Attention_Decoder(object):
    def __init__(self,seq_length,rnn_cell,loop_function,weight_decay = 0.,attn_vec_length = None, num_heads = 1):
        self.seq_length = seq_length
        self.cell = rnn_cell
        self.num_heads  = num_heads
        self.get_input = loop_function
        self.attn_vec_length = attn_vec_length
        self.weight_decay = weight_decay
        self.key_point = {}


    def get_variables(self):
        all_variables = tf.global_variables()
        self.train_variables = []
        self.train_variables_number = 0
        for v in all_variables:
            if (self.scope is not None and self.scope in v.name)\
                    or (self.scope is None and "Attention_Decoder" in v.name):
                self.train_variables.append(v)
                tmp_num = 1
                for n in v.shape:
                    tmp_num *= n.value
                self.train_variables_number += tmp_num
        return self.train_variables

    def ops(self,attention_state,is_training = None, scope = None, reuse = False):
        self.scope = scope
        self.attention_state = attention_state
        self.batch_size = self.attention_state.get_shape()[0].value
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
        self.out_size = self.cell.output_size
        with tf.variable_scope(self.scope,"Attention_Decoder",reuse = reuse):
            self.a_s_h = self.attention_state.get_shape()[1].value
            self.a_s_w = self.attention_state.get_shape()[2].value
            self.a_s_c = self.attention_state.get_shape()[3].value
            self.attention_sequence = tf.reshape(self.attention_state, [self.batch_size, self.a_s_w * self.a_s_h, 1, self.a_s_c])
            if self.attn_vec_length == None:
                self.attn_vec_length = self.a_s_c
            self.attns = []
            self.ds_s = []
            self.prev_states = []
            self.cell_out_s = []
            self.cell_input_s = []
            self.input_x_s = []
            prev_states = None
            prev_out = None
            res_outputs = []
            init_input_x = self.get_input(None, 0)
            input_x_length = init_input_x.get_shape()[1].value
            for i in xrange(self.seq_length):
                if i == 0:
                    prev_states = self.initial_state
                    attns,ds = self.build_attention(prev_states, False)
                    input_x = init_input_x
                else:
                    attns,ds = self.build_attention(prev_states, True)
                    input_x = self.get_input(prev_out,i)
                self.input_x_s .append(input_x)
                with tf.variable_scope("input_linear") as tmp_scope:
                    if i > 0 :
                        tmp_scope.reuse_variables()
                    cell_input = utils.linear([input_x] + ds, input_x_length,True)
                self.cell_input_s.append(cell_input)
                cell_state = prev_states
                with tf.variable_scope("run_cell") as tmp_scope:
                    if i > 0 :
                        tmp_scope.reuse_variables()
                cell_out, prev_states = self.cell(cell_input, cell_state)
                with tf.variable_scope("output_linear") as tmp_scope:
                    if i > 0 :
                        tmp_scope.reuse_variables()
                    attn_out = utils.linear([cell_out] + ds,self.out_size, True)
                prev_out = attn_out
                res_outputs.append(attn_out)
                self.attns.append(attns)
                self.ds_s.append(ds)
                self.prev_states.append(prev_states)
                self.cell_out_s.append(cell_out)
        self.key_point["attns"] = self.attns
        self.key_point["ds_s"] = self.ds_s
        self.key_point["prev_states"] = self.prev_states
        self.key_point["cell_out_s"] = self.cell_out_s
        self.key_point["cell_input_s"] = self.cell_input_s
        self.key_point["input_x_s"] = self.input_x_s
        self.get_variables()
        res_states = prev_states
        return res_outputs, res_states



    def build_attention(self,prev_states,reuse):
        # prev_state (2,batch_size,num_of_units)
        if nest.is_sequence(prev_states):  # If the query is a tuple, flatten it.
            state_list = nest.flatten(prev_states)
            for s in state_list:  # Check that ndims == 2 if specified.
                ndims = s.get_shape().ndims
                if ndims:
                    assert ndims == 2
            prev_state = tf.concat(prev_states, 1)
        attns = []
        ds = []
        for h in xrange(self.num_heads):
            with tf.variable_scope("attn_{0}_build".format(h),reuse = reuse):
                # e = V^T * tanh(S_t-1 * V_s + a_s*H + b)
                # S_t-1 * V_s
                S_V = utils.linear(prev_state,self.attn_vec_length,True)
                S_V = tf.reshape(S_V,[-1,1,1,self.attn_vec_length])
                # a_s*H
                h_conv_w = tf.get_variable("h_conv_w_{0}".format(h),[1, 1, self.a_s_c, self.attn_vec_length],)
                A_H = tf.nn.conv2d(self.attention_sequence,h_conv_w,[1,1,1,1],"SAME")
                b = tf.get_variable("b{0}".format(h),[1,1,self.attn_vec_length],initializer=tf.zeros_initializer)
                V_T = tf.get_variable("V_T{0}".format(h),[self.attn_vec_length])
                e = V_T * tf.tanh(S_V + A_H)
                e = tf.reduce_sum(e,axis=[2, 3])
                attn = tf.nn.softmax(e)
                d = tf.reduce_sum(tf.reshape(attn, [-1, self.a_s_w * self.a_s_h, 1, 1]) * self.attention_sequence, [1, 2])
                ds.append(d)
                attns.append(attn)
        return attns,ds





