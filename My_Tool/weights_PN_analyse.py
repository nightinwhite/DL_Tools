#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
parent_dir = "visualization_graphs/"
if os.path.exists(parent_dir) == False:
    os.mkdir(parent_dir)
weight_path = "/home/night/data/E119"
weight_names = ["I0_Attention_Model%Inception_v3%Conv2d_p1_3x3%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Conv2d_p2_3x3%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Conv2d_p3_3x3%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Conv2d_p5_1x1%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Conv2d_p6_3x3%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_0%a_conv2d%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_1%b_conv2d_1%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_1%b_conv2d_2%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_2%c_conv2d_1%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_2%c_conv2d_2%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_2%c_conv2d_3%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_2%c_conv2d_3%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_3%d_conv2d%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_0%a_conv2d%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_1%b_conv2d_1%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_1%b_conv2d_2%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_2%c_conv2d_1%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_2%c_conv2d_2%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_2%c_conv2d_3%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_3%d_conv2d%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_0%a_conv2d%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_1%b_conv2d_1%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_1%b_conv2d_2%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_2%c_conv2d_1%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_2%c_conv2d_2%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_2%c_conv2d_3%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_3%d_conv2d%Conv%weights:0",
                ]

save_path_all = os.path.join(parent_dir,"weights_PN_analyse/")
if os.path.exists(save_path_all) == False:
    os.mkdir(save_path_all)

for weight_name in weight_names:
    tmp_save_path = os.path.join(save_path_all,weight_name)
    if os.path.exists(tmp_save_path) == False:
        os.mkdir(tmp_save_path)

    tmp_weight_path = os.path.join(weight_path, weight_name+".npy")
    tmp_weight = np.load(tmp_weight_path)
    # 正负性
    for i in range(tmp_weight.shape[3]):
        print weight_name, i
        tmp_channel_weight = tmp_weight[:, :, :, i]
        tmp_channel_mask = tmp_channel_weight > 0
        tmp_channel_mask = np.asarray(tmp_channel_mask, np.int8)
        tmp_channel_weight_pos = tmp_channel_weight * tmp_channel_mask
        tmp_channel_weight_neg = tmp_channel_weight * (1 - tmp_channel_mask)
        tmp_channel_pos_num = np.sum(tmp_channel_mask)
        tmp_channel_neg_num = np.sum(1 - tmp_channel_mask)
        tmp_channel_pos_sum = np.sum(tmp_channel_weight_pos)
        tmp_channel_neg_sum = np.sum(tmp_channel_weight_neg)
        plt.subplot2grid((1, 2), (0, 0))
        plt.pie([tmp_channel_pos_num, tmp_channel_neg_num], colors=[(1, 0.2, 0.2), (0., 0.8, 1.)],
                labels=["pos:{0}".format(tmp_channel_pos_num), "neg:{0}".format(tmp_channel_neg_num)])
        plt.subplot2grid((1, 2), (0, 1))
        tmp_channel_pos_sum_rate = tmp_channel_pos_sum / (tmp_channel_pos_sum - tmp_channel_neg_sum)
        tmp_channel_neg_sum_rate = -tmp_channel_neg_sum / (tmp_channel_pos_sum - tmp_channel_neg_sum)
        plt.pie([tmp_channel_pos_sum_rate, tmp_channel_neg_sum_rate], colors=[(1, 0.2, 0.2), (0., 0.8, 1.)],
                labels=["pos:{0}".format(tmp_channel_pos_sum), "neg:{0}".format(tmp_channel_neg_sum)])
        tmp_save_name = os.path.join(tmp_save_path, "pos_neg_{0}.png".format(i))
        plt.savefig(tmp_save_name)
