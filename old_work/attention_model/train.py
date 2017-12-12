import tensorflow as tf
from DL_Tools.My_Data.TF_Reader import TF_Reader
from DL_Tools.My_Data.Data_Argument import *
from DL_Tools.My_Trainer import Trainer,Train_op
from DL_Tools.My_Trainer.Trainer_utils import *
from Attention_Model import Attention_Model
from tensorflow.python.platform import flags
import common_flags
import os
import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='0'
common_flags.define()
Flags = flags.FLAGS

arg_ops = [
    base_norm,
    distorting_color,
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
    train_tf_reader = TF_Reader(["/home/hp/data/syn_eng/tfrecord_new/train_1.records","/home/hp/data/syn_eng/tfrecord_new/train_2.records","/home/hp/data/syn_eng/tfrecord_new/train_3.records","/home/hp/data/syn_eng/tfrecord_new/train_4.records","/home/hp/data/syn_eng/tfrecord_new/train_5.records"],True)
    train_tf_reader.data_argument(arg_ops)
    train_images,train_labels = train_tf_reader.shuffle_batch()
with tf.variable_scope("val_tf_reader"):
    val_tf_reader = TF_Reader(["/home/hp/data/syn_eng/tfrecord_new/val.records"],True)
    val_tf_reader.data_argument(arg_ops)
    val_images,val_labels = val_tf_reader.shuffle_batch()
# with tf.variable_scope("visual_reader"):
#     visual_net_images = tf.placeholder(tf.float32, [Flags.batch_size, Flags.net_image_height, Flags.net_image_width,
#                                                     Flags.net_image_channel])
    # for ops in arg_ops:
    #     visual_net_images = ops(visual_net_images)
train_model = Attention_Model(train_images,train_labels)
train_model.build_model("Attention_Model")
train_model_pred_res = train_model.pred_res

val_model = Attention_Model(val_images)
val_model.is_training = False
val_model.build_model("Attention_Model",True)
val_model_pred_res = val_model.pred_res

# visual_model = Attention_Model(visual_net_images)
# visual_model.is_training = False
# visual_model.build_model("Attention_Model", True)
# visual_model_pred_res = visual_model.pred_res

train_loss = train_model.get_loss(train_labels)
val_loss = val_model.get_loss(val_labels)

train_single_acc = single_accuracy(train_model_pred_res,train_labels)
train_seq_acc = seq_accuracy(train_model_pred_res,train_labels)
val_single_acc = single_accuracy(val_model_pred_res,val_labels)
val_seq_acc = seq_accuracy(val_model_pred_res,val_labels)

epoch = Flags.epoch
iteration_in_epoch  = Flags.train_iter_num
iteration_in_val = Flags.val_iter_num

train_op = Train_op.Train_op("MomentumOptimizer", Flags.learning_rate, clip_value=Flags.clip_gradient_norm).build(train_loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess)
# tf.summary.image("train_images",train_images,10)
# tf.summary.scalar("train_loss",train_loss)
# tf.summary.scalar("train_single_acc",train_single_acc)
# tf.summary.scalar("train_seq_acc",train_seq_acc)
# merge_summary = tf.summary.merge_all()
# summary_writer = tf.summary.FileWriter("train_log/",sess.graph)

my_trainer = Trainer.Trainer(sess = sess,
                             train_op = train_op,
                             epoch = Flags.epoch,
                             train_iter_num = Flags.train_iter_num,
                             val_iter_num = Flags.val_iter_num
                             )
my_trainer.add_train_info_ops(
    Trainer.Info_Ops(train_loss, "train_loss", 1))
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

# visual_weights:
all_variables = tf.global_variables()
tmp_count = 0
for v in all_variables:
    if "Momentum" not in v.name:
        tmp_count += 1
        my_trainer.add_train_info_ops(Trainer.Info_Ops(v,v.name.replace("/","%"),1000,visible=False,need_to_be_saved=True))
# graph = tf.get_default_graph()
# for key in graph.get_all_collection_keys():
#     if key != 'variables'and key!='trainable_variables' \
#             and key != 'regularization_losses'and key!='queue_runners' \
#             and key != 'summaries'and key!='global_step' \
#             and key != 'cond_context'and key!='train_op' \
#             and key != 'update_ops'and key!='model_variables':
#         print key
#         my_trainer.add_train_info_ops(Trainer.Info_Ops(tf.get_collection(key), key, 10,visible=False))
#         my_trainer.add_visual_info(key)

my_trainer.set_model_saving_info("models/","AN_cap2")
# load value
def cmp_value(bef_value, now_value):
    if bef_value < now_value:
        return True
    else:
        return False

my_trainer.set_best_saving_info("val_seq_acc",cmp_fuc=cmp_value)

# train_model.restore_model(sess, "models/AN_cap2_new")

# visual_real_images = []
# visual_data_path = "visual_data/"
# visual_names = os.listdir(visual_data_path)
# for n in visual_names[:Flags.batch_size]:
#     tmp_img = cv2.imread(os.path.join(visual_data_path,n))
#     tmp_img = cv2.resize(tmp_img,(Flags.net_image_width, Flags.net_image_height))
#     tmp_img = np.asarray(tmp_img,np.float32)
#     tmp_img = tmp_img / 255. - 0.5
#     tmp_img = tmp_img*1.5
#     visual_real_images.append(tmp_img)
# visual_real_images = np.asarray(visual_real_images)
# def visual_save_fuc(res_data,e):
#     if os.path.exists("visual_save_path/"):
#         os.mkdir("visual_save_path")
#     np.save("visual_save_path/e_{0}_.npy".format(e),res_data)
#
# my_trainer.add_visual_info_ops(visual_model_pred_res,visual_net_images,visual_real_images,visual_save_fuc)

my_trainer.run()
