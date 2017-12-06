import tensorflow as tf
from My_Data.TF_Reader import TF_Reader
from My_Data.Data_Argument import *
from My_Trainer import Trainer
from Attention_Model import Attention_Model
from tensorflow.python.platform import flags
import common_flags
import My_Log.Log_Manager as LM
import tensorflow.contrib.slim as slim
import sys
import os
import cv2
import numpy as np
from My_Tool.visual_area import *

common_flags.define()
Flages = flags.FLAGS

arg_ops = [
    base_norm,
    # distorting_color
]

test_images = tf.placeholder(tf.float32,[Flages.batch_size,Flages.net_image_height, Flages.net_image_width, Flages.net_image_channel])

test_model = Attention_Model(test_images)
test_model.is_training = False
test_model.build_model("Attention_Model")
test_model_pred_res = test_model.pred_res

test_path = "/home/night/data/cap3/test/"
test_names = os.listdir(test_path)
real_labels = []
real_imgs = []
for n in test_names:
    real_img_path = os.path.join(test_path, n)
    real_img = cv2.imread(real_img_path)
    real_label = n.split(".")[0]
    real_label = real_label.split("_")[1]
    real_imgs.append(real_img)
    real_labels.append(real_label)

real_imgs = np.asarray(real_imgs,np.float32)
real_imgs = real_imgs/255. - 0.5

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
test_model.restore_model(sess, "models/attention_model_Momentum_val_seq_acc_0.93125")

label_to_char = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#"
res,attns = sess.run([test_model_pred_res,test_model.get_attns()], feed_dict={test_images: real_imgs[:Flages.batch_size]})
print res.shape
attns = np.asarray(attns)
print attns.shape
res = np.argmax(res,-1)
res_char = []
for l in res:
    tmp_chars = ""
    for label in l:
        tmp_chars += label_to_char[label]
    res_char.append(tmp_chars)

res_char = res_char[:Flages.batch_size]
index_test = 7
test_attns = attns[:,0,index_test,:]
print test_attns.shape
test_attns = np.reshape(test_attns,[Flages.label_length,6,22])

def atten_map(attn):
    # p8.c
    map_p8_c_0 = single_map_value(attn, [1, 1], [0, 0], [1, 1])
    map_p8_c_1 = single_map_value(attn, [5, 5], [2, 2], [1, 1])
    map_p8_c_1 = single_map_value(map_p8_c_1, [1, 1], [0, 0], [1, 1])
    map_p8_c_2 = single_map_value(attn, [3, 3], [1, 1], [1, 1])
    map_p8_c_2 = single_map_value(map_p8_c_2, [3, 3], [1, 1], [1, 1])
    map_p8_c_2 = single_map_value(map_p8_c_2, [1, 1], [0, 0], [1, 1])
    map_p8_c_3 = single_map_value(attn, [1, 1], [0, 0], [1, 1])
    map_p8_c_3 = single_map_value(map_p8_c_3, [3, 3], [1, 1], [1, 1])
    map_p8_c = 64./288.*map_p8_c_0 + \
               64./288.*map_p8_c_1 + \
               96./288.*map_p8_c_2 + \
               64./288.*map_p8_c_3
    # p8.b
    map_p8_b_0 = single_map_value(map_p8_c, [1, 1], [0, 0], [1, 1])
    map_p8_b_1 = single_map_value(map_p8_c, [5, 5], [2, 2], [1, 1])
    map_p8_b_1 = single_map_value(map_p8_b_1, [1, 1], [0, 0], [1, 1])
    map_p8_b_2 = single_map_value(map_p8_c, [3, 3], [1, 1], [1, 1])
    map_p8_b_2 = single_map_value(map_p8_b_2, [3, 3], [1, 1], [1, 1])
    map_p8_b_2 = single_map_value(map_p8_b_2, [1, 1], [0, 0], [1, 1])
    map_p8_b_3 = single_map_value(map_p8_c, [1, 1], [0, 0], [1, 1])
    map_p8_b_3 = single_map_value(map_p8_b_3, [3, 3], [1, 1], [1, 1])
    map_p8_b = 64. / 288. * map_p8_b_0 + \
               64. / 288. * map_p8_b_1 + \
               96. / 288. * map_p8_b_2 + \
               64. / 288. * map_p8_b_3
    # p8.a
    map_p8_a_0 = single_map_value(map_p8_b, [1, 1], [0, 0], [1, 1])
    map_p8_a_1 = single_map_value(map_p8_b, [5, 5], [2, 2], [1, 1])
    map_p8_a_1 = single_map_value(map_p8_a_1, [1, 1], [0, 0], [1, 1])
    map_p8_a_2 = single_map_value(map_p8_b, [3, 3], [1, 1], [1, 1])
    map_p8_a_2 = single_map_value(map_p8_a_2, [3, 3], [1, 1], [1, 1])
    map_p8_a_2 = single_map_value(map_p8_a_2, [1, 1], [0, 0], [1, 1])
    map_p8_a_3 = single_map_value(map_p8_b, [1, 1], [0, 0], [1, 1])
    map_p8_a_3 = single_map_value(map_p8_a_3, [3, 3], [1, 1], [1, 1])
    map_p8_a = 64. / 288. * map_p8_a_0 + \
               64. / 288. * map_p8_a_1 + \
               96. / 288. * map_p8_a_2 + \
               32. / 288. * map_p8_a_3
    # p7
    map_p7 = single_map_value(map_p8_a, [3, 3], [0, 0], [2, 2])
    map_p6 = single_map_value(map_p7, [3, 3], [0, 0], [1, 1])
    map_p5 = single_map_value(map_p6, [1, 1], [0, 0], [1, 1])
    map_p4 = single_map_value(map_p5, [3, 3], [0, 0], [2, 2])
    map_p3 = single_map_value(map_p4, [3, 3], [1, 1], [1, 1])
    map_p2 = single_map_value(map_p3, [3, 3], [0, 0], [1, 1])
    map_p1 = single_map_value(map_p2, [3, 3], [0, 0], [2, 2])

    return map_p1

res_map_0 = atten_map(test_attns[1])

def transform_to_image(res_map):
    res_map = (res_map - np.min(res_map))/(np.max(res_map) - np.min(res_map))*255.
    res_map = np.asarray(res_map,np.uint8)
    return res_map

# save_path = "attention_img/cap1_attention"
save_path = "attention_img/cap3_attention_by_cap2"
if os.path.exists(save_path) == False:
    os.mkdir(save_path)

for i,test_attn in enumerate(test_attns):
    res_map = atten_map(test_attn)
    res_map = transform_to_image(res_map)
    res_mask = np.greater(res_map,0)
    res_mask = np.asarray(res_mask,np.uint8)
    res_map = res_map * res_mask
    res_map = [np.expand_dims(res_map,2),np.expand_dims(res_map,2),np.expand_dims(res_map,2)]
    res_map = np.concatenate(res_map,2)
    res_img = real_imgs[index_test]
    res_img = res_img[:res_map.shape[0],:res_map.shape[1],:]
    res_img = transform_to_image(res_img)
    mix_res_map = cv2.addWeighted(res_map,0.6,res_img,0.4,0)
    cv2.imshow("{0}".format(i), mix_res_map)
    cv2.imwrite(os.path.join(save_path,"{0}_{1}_{2}.png".format(i,res_char[index_test],real_labels[index_test])),mix_res_map)

cv2.waitKey(0)
cv2.destroyAllWindows()