# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

parent_dir = "visualization_graphs/"
if os.path.exists(parent_dir) == False:
    os.mkdir(parent_dir)

weight_aft_path = "/home/night/data/E119"
weight_bef_path = "/home/night/data/E36"
weight_names = ["I0_Attention_Model%Inception_v3%Conv2d_p1_3x3%Conv%weights:0",
                "I0_Attention_Model%Inception_v3%Conv2d_p2_3x3%Conv%weights:0"]
save_path_all = os.path.join(parent_dir, "gradient_heat_analyse")
if os.path.exists(save_path_all) == False:
    os.mkdir(save_path_all)

for weight_name in weight_names:
    tmp_save_path = os.path.join(save_path_all, weight_name)
    if os.path.exists(tmp_save_path) == False:
        os.mkdir(tmp_save_path)

    tmp_bef_weight_path = os.path.join(weight_bef_path, weight_name + ".npy")
    tmp_bef_weight = np.load(tmp_bef_weight_path)
    tmp_aft_weight_path = os.path.join(weight_aft_path, weight_name + ".npy")
    tmp_aft_weight = np.load(tmp_aft_weight_path)
    tmp_weight = tmp_aft_weight - tmp_bef_weight
    # 梯度正负性
    for i in range(tmp_weight.shape[3]):
        tmp_channel_weight = np.reshape(tmp_weight[:, :, :, i],
                                        (tmp_weight.shape[0] * tmp_weight.shape[1], tmp_weight.shape[2]))
        print weight_name, i
        sns.heatmap(tmp_channel_weight, center=0)
        tmp_save_name = os.path.join(tmp_save_path, "heat_map_{0}.png".format(i))
        fig = plt.gcf()
        fig.savefig(tmp_save_name)
        plt.clf()