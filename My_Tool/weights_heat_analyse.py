#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
parent_dir = "visualization_graphs/"
if os.path.exists(parent_dir) == False:
    os.mkdir(parent_dir)
weight_path = "/home/night/data/visual_res"
# weight_names = ["I0_Attention_Model%Inception_v3%Conv2d_p1_3x3%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Conv2d_p2_3x3%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Conv2d_p3_3x3%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Conv2d_p5_1x1%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Conv2d_p6_3x3%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_0%a_conv2d%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_1%b_conv2d_1%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_1%b_conv2d_2%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_2%c_conv2d_1%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_2%c_conv2d_2%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_2%c_conv2d_3%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_2%c_conv2d_3%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_a%Branch_3%d_conv2d%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_0%a_conv2d%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_1%b_conv2d_1%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_1%b_conv2d_2%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_2%c_conv2d_1%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_2%c_conv2d_2%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_2%c_conv2d_3%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_b%Branch_3%d_conv2d%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_0%a_conv2d%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_1%b_conv2d_1%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_1%b_conv2d_2%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_2%c_conv2d_1%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_2%c_conv2d_2%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_2%c_conv2d_3%Conv%weights:0",
#                 "I0_Attention_Model%Inception_v3%Mixed_p8_c%Branch_3%d_conv2d%Conv%weights:0",
#                 ]
weight_names = ["AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_1a_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_2a_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_2b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_3b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Conv2d_4a_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_1%Conv2d_0b_5x5%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_2%Conv2d_0b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_2%Conv2d_0c_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5b%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_1%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_1%Conv_1_0c_5x5%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_2%Conv2d_0b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_2%Conv2d_0c_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5c%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_1%Conv2d_0b_5x5%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_2%Conv2d_0b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_2%Conv2d_0c_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_5d%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6a%Branch_0%Conv2d_1a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6a%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6a%Branch_1%Conv2d_0b_3x3%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6a%Branch_1%Conv2d_1a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_1%Conv2d_0b_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_1%Conv2d_0c_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0b_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0c_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0d_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_2%Conv2d_0e_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6b%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_1%Conv2d_0b_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_1%Conv2d_0c_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0b_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0c_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0d_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_2%Conv2d_0e_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6c%Branch_3%Conv2d_0b_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_0%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_1%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_1%Conv2d_0b_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_1%Conv2d_0c_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0a_1x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0b_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0c_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0d_7x1%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_2%Conv2d_0e_1x7%weights:0",
"AttentionOcr%conv_tower_fn%INCE%InceptionV3%Mixed_6d%Branch_3%Conv2d_0b_1x1%weights:0",]

save_path_all = os.path.join(parent_dir,"weights_heat_analyse/")
if os.path.exists(save_path_all) == False:
    os.mkdir(save_path_all)

for weight_name in weight_names:
    tmp_save_path = os.path.join(save_path_all,weight_name)
    if os.path.exists(tmp_save_path) == False:
        os.mkdir(tmp_save_path)

    tmp_weight_path = os.path.join(weight_path, weight_name+".npy")
    tmp_weight = np.load(tmp_weight_path)
    # print weight_name,tmp_weight.shape
    for i in range(tmp_weight.shape[3]):
        print weight_name, i
        tmp_channel_weight = np.reshape(tmp_weight[:, :, :, i],
                                        (tmp_weight.shape[0] * tmp_weight.shape[1], tmp_weight.shape[2]))
        sns.heatmap(tmp_channel_weight, center=0)
        tmp_save_name = os.path.join(tmp_save_path, "heat_map_{0}.png".format(i))
        fig = plt.gcf()
        fig.savefig(tmp_save_name)
        plt.clf()
