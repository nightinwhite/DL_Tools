#coding:utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import common_flags
from tensorflow.python.platform import flags as flags
from chinese_encoder import Chinese_Map_NN
from My_NN.My_Trainer import Trainer
import read_picture
import cv2
import os
import numpy as np

FLAGS = flags.FLAGS
common_flags.define()

net_train_images = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.net_img_height,FLAGS.net_img_width,FLAGS.net_img_channel],name="net_train_images")
net_train_images_class = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.net_img_height,FLAGS.net_img_width,FLAGS.net_img_channel],name="net_train_images_class")
net_train_images_style = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.net_img_height,FLAGS.net_img_width,FLAGS.net_img_channel],name="net_train_images_style")

net_val_images = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.net_img_height,FLAGS.net_img_width,FLAGS.net_img_channel],name="net_val_images")
net_val_images_class = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.net_img_height,FLAGS.net_img_width,FLAGS.net_img_channel],name="net_val_images_class")
net_val_images_style = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.net_img_height,FLAGS.net_img_width,FLAGS.net_img_channel],name="net_val_images_style")

net_visual_images = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.net_img_height,FLAGS.net_img_width,FLAGS.net_img_channel],name="net_visual_images")
#--------------------------------------------------------
net_train_images = net_train_images / 255. - 0.5
net_train_images_class = net_train_images_class / 255. - 0.5
net_train_images_style = net_train_images_style / 255. - 0.5

net_val_images = net_val_images / 255. - 0.5
net_val_images_class = net_val_images_class / 255. - 0.5
net_val_images_style = net_val_images_style / 255. - 0.5

net_visual_images = net_visual_images / 255. - 0.5
#--------------------------------------------------------
net_train_model = Chinese_Map_NN(net_train_images,True)
net_train_model.build_model()

net_train_model_class = Chinese_Map_NN(net_train_images_class,True)
net_train_model_class.build_model(reuse=True)

net_train_model_style = Chinese_Map_NN(net_train_images_style,True)
net_train_model_style.build_model(reuse=True)
#--------------------------------------------------------
net_val_model = Chinese_Map_NN(net_val_images,False)
net_val_model.build_model(reuse=True)

net_val_model_class = Chinese_Map_NN(net_val_images_class,False)
net_val_model_class.build_model(reuse=True)

net_val_model_style = Chinese_Map_NN(net_val_images_style,False)
net_val_model_style.build_model(reuse=True)
#--------------------------------------------------------
net_visual_model = Chinese_Map_NN(net_visual_images,False)
net_visual_model.build_model(reuse=True)

with tf.variable_scope("Build_Train_Losses"):
    with tf.variable_scope("map_loss"):
        train_map_loss = tf.reduce_mean(tf.square(net_train_model.map_img - net_train_images) +
                                        tf.square(net_train_model_class.map_img - net_train_images_class) +
                                        tf.square(net_train_model_style.map_img - net_train_images_style))
    with tf.variable_scope("class_loss"):
        train_class_loss = tf.reduce_mean(tf.square(net_train_model.class_vector - net_train_model_class.class_vector)-
                                          tf.square(net_train_model.class_vector - net_train_model_style.class_vector))#是否用交叉熵

    with tf.variable_scope("style_loss"):
        train_style_loss = tf.reduce_mean(tf.square(net_train_model.style_vector - net_train_model_style.style_vector))#是否用交叉熵

    train_loss = FLAGS.map_factor * train_map_loss + \
                 FLAGS.class_factor * train_class_loss + \
                 FLAGS.style_factor * train_style_loss

with tf.variable_scope("Build_Val_Losses"):
    with tf.variable_scope("map_loss"):
        val_map_loss = tf.reduce_mean(tf.square(net_val_model.map_img - net_val_images) +
                                        tf.square(net_val_model_class.map_img - net_val_images_class) +
                                        tf.square(net_val_model_style.map_img - net_val_images_style))
    with tf.variable_scope("class_loss"):
        val_class_loss = tf.reduce_mean(
            tf.square(net_val_model.class_vector - net_val_model_class.class_vector)-
            tf.square(net_val_model.class_vector - net_val_model_style.class_vector))  # 是否用交叉熵

    with tf.variable_scope("style_loss"):
        val_style_loss = tf.reduce_mean(
            tf.square(net_val_model.style_vector - net_val_model_style.style_vector))  # 是否用交叉熵

    val_loss = FLAGS.map_factor * val_map_loss + \
                 FLAGS.class_factor * val_class_loss + \
                 FLAGS.style_factor * val_style_loss

epoch = FLAGS.epoch
iteration_in_epoch  = FLAGS.train_iter_num
iteration_in_val = FLAGS.val_iter_num

sess = tf.InteractiveSession()

# summarys:
all_variables = tf.global_variables()
for v in all_variables:
    tf.summary.histogram(v.name,v)
tf.summary.image("train_images",net_train_images,10)
tf.summary.image("train_images_class",net_train_images_class,10)
tf.summary.image("train_images_style",net_train_images_style,10)

tf.summary.scalar("train_map_loss",train_map_loss)
tf.summary.scalar("train_class_loss",train_class_loss)
tf.summary.scalar("train_style_loss",train_style_loss)

merge_summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("train_log/",sess.graph)

my_trainer = Trainer.Trainer(sess = sess,
                             train_loss = train_loss,
                             optimizer_name = 'MomentumOptimizer',
                             learning_rate = FLAGS.learning_rate,
                             epoch = FLAGS.epoch,
                             train_iter_num = FLAGS.train_iter_num,
                             val_iter_num = FLAGS.val_iter_num
                             )
my_trainer.set_clip_value(FLAGS.clip_gradient_norm)

my_trainer.add_train_info_ops(
    Trainer.Info_Ops(train_class_loss,"train_class_loss",1))
my_trainer.add_train_info_ops(
    Trainer.Info_Ops(train_style_loss, "train_style_loss", 1))

my_trainer.add_val_info_ops(
    Trainer.Info_Ops(val_loss, "val_loss", 1))
my_trainer.add_val_info_ops(
    Trainer.Info_Ops(val_class_loss, "val_class_loss", 1))
my_trainer.add_val_info_ops(
    Trainer.Info_Ops(val_style_loss, "val_style_loss", 1))

my_trainer.set_save_tf_info("models/","Chinese_map_2048")

def cmp_value(bef_value, now_value):
    if bef_value < now_value:
        return True
    else:
        return False

my_trainer.set_best_save_info("val_loss",cmp_fuc=cmp_value)

my_trainer.set_summary_writer(merge_summary, summary_writer, 10)

my_trainer.init_model()



saver = tf.train.Saver(max_to_keep=3)
saver.restore(sess, "models/Chinese_map_2048_new")

my_trainer.add_visual_info("train_loss")
my_trainer.add_visual_info("train_class_loss")
my_trainer.add_visual_info("train_style_loss")
my_trainer.add_visual_info("val_loss")
my_trainer.add_visual_info("val_class_loss")
my_trainer.add_visual_info("val_style_loss")

train_rd_picture = read_picture.Image_Reader(FLAGS.batch_size,FLAGS.batch_size*4,4,"/home/night/data/chinese_font/font_true_info/","/home/night/data/chinese_font/font_imgs/")
def get_train_feed_dict():
    res = train_rd_picture.get_batch_data(FLAGS.batch_size)
    return {net_train_images:res[:,0,:,:,:],net_train_images_class:res[:,1,:,:,:],net_train_images_style:res[:,2,:,:,:],}

val_rd_picture = read_picture.Image_Reader(FLAGS.batch_size,FLAGS.batch_size*4,4,"font_info/","font_imgs/")
def get_val_feed_dict():
    res = val_rd_picture.get_batch_data(FLAGS.batch_size)
    return {net_val_images:res[:,0,:,:,:],net_val_images_class:res[:,1,:,:,:],net_val_images_style:res[:,2,:,:,:],}
my_trainer.set_get_train_feed_dict(get_train_feed_dict)
my_trainer.set_get_val_feed_dict(get_val_feed_dict)

test_real_data = []
test_data_path = "test_data/"
test_img_names = os.listdir(test_data_path)
for test_img_name in test_img_names:
    tmp_img = cv2.imread(test_data_path+test_img_name)
    tmp_img = tmp_img/255.
    test_real_data.append(tmp_img)
test_real_data = np.asarray(test_real_data)
test_real_data = test_real_data[:FLAGS.batch_size]
print test_real_data.shape
def visual_save_fuc(res_data,e):
    save_path = "visual_data_2/{0}/".format(e+178)
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    for i,tmp_img in enumerate(res_data):
        tmp_img = (tmp_img - np.min(tmp_img))/(np.max(tmp_img) - np.min(tmp_img))*255.
        tmp_img = np.asarray(tmp_img,np.uint8)
        cv2.imwrite(save_path+"{0}.png".format(i),tmp_img)
my_trainer.add_visual_info_ops(net_visual_model.map_img,net_visual_images,test_real_data,visual_save_fuc)
my_trainer.run()