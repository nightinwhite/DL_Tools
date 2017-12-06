import tensorflow as tf
from My_Data.TF_Reader import TF_Reader
from My_Data.Data_Argument import *
from My_Trainer import Trainer
from My_Trainer.Trainer_utils import *
from Attention_Model import Attention_Model
from tensorflow.python.platform import flags
import common_flags
import My_Log.Log_Manager as LM
import tensorflow.contrib.slim as slim
import sys
import os
import cv2
import numpy as np

common_flags.define()
Flages = flags.FLAGS

arg_ops = [
    base_norm,
    # distorting_color
]

# tmp_path = "/media/night/0002FCA800053168/data/fsns_tf_train/"
# tmp_files_name = os.listdir(tmp_path)
# train_files_name = []
# for name in tmp_files_name:
#     train_files_name.append(os.path.join(tmp_path,name))
#
# tmp_path = "/media/night/0002FCA800053168/data/fsns_tf_validation/"
# tmp_files_name = os.listdir(tmp_path)
# val_files_name = []
# for name in tmp_files_name:
#     val_files_name.append(os.path.join(tmp_path,name))

with tf.variable_scope("train_tf_reader"):
    train_tf_reader = TF_Reader(["tfrecords/cap2_train.record"],True)
    train_tf_reader.data_argument(arg_ops)
    train_images,train_labels = train_tf_reader.shuffle_batch()
with tf.variable_scope("val_tf_reader"):
    val_tf_reader = TF_Reader(["tfrecords/cap2_val.record"],True)
    val_tf_reader.data_argument(arg_ops)
    val_images,val_labels = val_tf_reader.shuffle_batch()
with tf.variable_scope("visual_reader"):
    visual_net_images = tf.placeholder(tf.float32, [Flages.batch_size, Flages.net_image_height, Flages.net_image_width,
                                                    Flages.net_image_channel])
    for ops in arg_ops:
        visual_net_images = ops(visual_net_images)
train_model = Attention_Model(train_images,train_labels)
train_model.build_model("Attention_Model")
train_model_pred_res = train_model.pred_res

val_model = Attention_Model(val_images)
val_model.is_training = False
val_model.build_model("Attention_Model",True)
val_model_pred_res = val_model.pred_res

visual_model = Attention_Model(visual_net_images)
visual_model.is_training = False
visual_model.build_model("Attention_Model", True)
visual_model_pred_res = visual_model.pred_res

train_loss = train_model.get_loss(train_labels)
val_loss = val_model.get_loss(val_labels)

train_single_acc = single_accuracy(train_model_pred_res,train_labels)
train_seq_acc = seq_accuracy(train_model_pred_res,train_labels)
val_single_acc = single_accuracy(val_model_pred_res,val_labels)
val_seq_acc = seq_accuracy(val_model_pred_res,val_labels)

epoch = Flages.epoch
iteration_in_epoch  = Flages.train_iter_num
iteration_in_val = Flages.val_iter_num

sess = tf.InteractiveSession()

# summarys:
all_variables = tf.global_variables()
for v in all_variables:
    # print v.name
    tf.summary.histogram(v.name,v)
tf.summary.image("train_images",train_images,10)
tf.summary.scalar("train_loss",train_loss)
tf.summary.scalar("train_single_acc",train_single_acc)
tf.summary.scalar("train_seq_acc",train_seq_acc)
merge_summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("train_log/",sess.graph)

my_trainer = Trainer.Trainer(sess = sess,
                             train_loss = train_loss,
                             optimizer_name = 'MomentumOptimizer',
                             learning_rate = Flages.learning_rate,
                             epoch = Flages.epoch,
                             train_iter_num = Flages.train_iter_num,
                             val_iter_num = Flages.val_iter_num
                             )
my_trainer.set_clip_value(Flages.clip_gradient_norm)

my_trainer.add_train_info_ops(
    Trainer.Info_Ops(train_single_acc,"train_single_acc",1))
my_trainer.add_train_info_ops(
    Trainer.Info_Ops(train_seq_acc, "train_seq_acc", 1))

my_trainer.add_val_info_ops(
    Trainer.Info_Ops(val_loss, "val_loss", 1))
my_trainer.add_val_info_ops(
    Trainer.Info_Ops(val_single_acc, "val_single_acc", 1))
my_trainer.add_val_info_ops(
    Trainer.Info_Ops(val_seq_acc, "val_seq_acc", 1))

my_trainer.set_save_tf_info("models/","AN_cap2")

def cmp_value(bef_value, now_value):
    if bef_value < now_value:
        return True
    else:
        return False

my_trainer.set_best_save_info("val_seq_acc",cmp_fuc=cmp_value)

my_trainer.set_summary_writer(merge_summary, summary_writer, 10)

my_trainer.init_model()

# train_model.restore_model(sess, "models/AN_cap2_new")

my_trainer.add_visual_info("train_loss")
my_trainer.add_visual_info("train_single_acc")
my_trainer.add_visual_info("val_loss")
my_trainer.add_visual_info("val_single_acc")

visual_real_images = []
visual_data_path = "visual_data/"
visual_names = os.listdir(visual_data_path)
for n in visual_names[:Flages.batch_size]:
    tmp_img = cv2.imread(os.path.join(visual_data_path,n))
    visual_real_images.append(tmp_img)
visual_real_images = np.asarray(visual_real_images)
def visual_save_fuc(res_data,e):
    np.save("visual_save_path/e_{0}_.npy".format(e),res_data)

my_trainer.add_visual_info_ops(visual_model_pred_res,visual_net_images,visual_real_images,visual_save_fuc)

my_trainer.run()
