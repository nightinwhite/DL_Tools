#coding:utf-8
import tensorflow.contrib.slim as slim
import tensorflow as tf
class Train_op(object):
    def __init__(self,optimizer_name,lr,momentum = 0.9,clip_value = 0):
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.momentum = momentum
        self.clip_value = clip_value

    def build(self,loss):
        if self.optimizer_name == "MomentumOptimizer":
            self.optimizer = tf.train.MomentumOptimizer(self.lr,self.momentum)
        if self.optimizer_name == "AdamOptimizer":
            self.optimizer = tf.train.AdamOptimizer(self.lr)

        if self.clip_value != 0:
            with tf.variable_scope("clip_gradient_norm"):
                train_op = slim.learning.create_train_op(loss, self.optimizer,
                                                  clip_gradient_norm=self.clip_value)
        else:
            train_op = self.optimizer.minimize(loss)
        return train_op
