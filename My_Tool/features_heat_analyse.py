#coding:utf-8
from My_Tool.feature_extract_visualization import *
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

l1 = Feature_Extract_Visualization_Layer(None,(3,3),"VALID",(2,2),None)
l2 = Feature_Extract_Visualization_Layer(None,(3,3),"VALID",(1,1),[l1])
l3 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME", (1,1),[l2])
l4 = Feature_Extract_Visualization_Layer(None,(3,3),"VALID",(2,2),[l3])
l5 = Feature_Extract_Visualization_Layer(None,(1,1),"VALID",(1,1),[l4])
l6 = Feature_Extract_Visualization_Layer(None,(3,3),"VALID",(1,1),[l5])
l7 = Feature_Extract_Visualization_Layer(None,(3,3),"VALID",(2,2),[l6])
#mixed 5b
l8_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l7])
l8_1_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l7])
l8_1_1 = Feature_Extract_Visualization_Layer(None,(5,5),"SAME",(1,1),[l8_1_0])
l8_2_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l7])
l8_2_1 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME",(1,1),[l8_2_0])
l8_2_2 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME",(1,1),[l8_2_1])
l8_3_0 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME",(1,1),[l7])
l8_3_1 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l8_3_0])
#合并
l8 =  Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l8_0,l8_1_1,l8_2_2,l8_3_1])
#mixed 5c
l9_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l8])
l9_1_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l8])
l9_1_1 = Feature_Extract_Visualization_Layer(None,(5,5),"SAME",(1,1),[l9_1_0])
l9_2_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l8])
l9_2_1 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME",(1,1),[l9_2_0])
l9_2_2 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME",(1,1),[l9_2_1])
l9_3_0 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME",(1,1),[l8])
l9_3_1 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l9_3_0])
#合并
l9 =  Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l9_0,l9_1_1,l9_2_2,l9_3_1])
#mixed 5d
l10_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l9])
l10_1_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l9])
l10_1_1 = Feature_Extract_Visualization_Layer(None,(5,5),"SAME",(1,1),[l10_1_0])
l10_2_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l9])
l10_2_1 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME",(1,1),[l10_2_0])
l10_2_2 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME",(1,1),[l10_2_1])
l10_3_0 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME",(1,1),[l9])
l10_3_1 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l10_3_0])
#合并
l10 =  Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l10_0,l10_1_1,l10_2_2,l10_3_1])
#mixed 6a
l11_0 = Feature_Extract_Visualization_Layer(None,(3,3),"VALID",(2,2),[l10])
l11_1_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l10])
l11_1_1 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME",(1,1),[l11_1_0])
l11_1_2 = Feature_Extract_Visualization_Layer(None,(3,3),"VALID",(2,2),[l11_1_1])
l11_2 = Feature_Extract_Visualization_Layer(None,(3,3),"VALID",(2,2),[l10])
#合并
l11 =  Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l11_0,l11_1_2,l11_2])
#mixed 6b
l12_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l11])
l12_1_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l11])
l12_1_1 = Feature_Extract_Visualization_Layer(None,(1,7),"SAME",(1,1),[l12_1_0])
l12_1_2 = Feature_Extract_Visualization_Layer(None,(7,1),"SAME",(1,1),[l12_1_1])
l12_2_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l11])
l12_2_1 = Feature_Extract_Visualization_Layer(None,(1,7),"SAME",(1,1),[l12_2_0])
l12_2_2 = Feature_Extract_Visualization_Layer(None,(7,1),"SAME",(1,1),[l12_2_1])
l12_2_3 = Feature_Extract_Visualization_Layer(None,(1,7),"SAME",(1,1),[l12_2_2])
l12_2_4 = Feature_Extract_Visualization_Layer(None,(7,1),"SAME",(1,1),[l12_2_3])
l12_3_0 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME",(1,1),[l11])
l12_3_1 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l12_3_0])
#合并
l12 =  Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l12_0,l12_1_2,l12_2_4,l12_3_1])
#mixed 6c
l13_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l12])
l13_1_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l12])
l13_1_1 = Feature_Extract_Visualization_Layer(None,(1,7),"SAME",(1,1),[l13_1_0])
l13_1_2 = Feature_Extract_Visualization_Layer(None,(7,1),"SAME",(1,1),[l13_1_1])
l13_2_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l12])
l13_2_1 = Feature_Extract_Visualization_Layer(None,(1,7),"SAME",(1,1),[l13_2_0])
l13_2_2 = Feature_Extract_Visualization_Layer(None,(7,1),"SAME",(1,1),[l13_2_1])
l13_2_3 = Feature_Extract_Visualization_Layer(None,(1,7),"SAME",(1,1),[l13_2_2])
l13_2_4 = Feature_Extract_Visualization_Layer(None,(7,1),"SAME",(1,1),[l13_2_3])
l13_3_0 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME",(1,1),[l12])
l13_3_1 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l13_3_0])
#合并
l13 =  Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l13_0,l13_1_2,l13_2_4,l13_3_1])
#mixed 6d
l14_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l13])
l14_1_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l13])
l14_1_1 = Feature_Extract_Visualization_Layer(None,(1,7),"SAME",(1,1),[l14_1_0])
l14_1_2 = Feature_Extract_Visualization_Layer(None,(7,1),"SAME",(1,1),[l14_1_1])
l14_2_0 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l13])
l14_2_1 = Feature_Extract_Visualization_Layer(None,(1,7),"SAME",(1,1),[l14_2_0])
l14_2_2 = Feature_Extract_Visualization_Layer(None,(7,1),"SAME",(1,1),[l14_2_1])
l14_2_3 = Feature_Extract_Visualization_Layer(None,(1,7),"SAME",(1,1),[l14_2_2])
l14_2_4 = Feature_Extract_Visualization_Layer(None,(7,1),"SAME",(1,1),[l14_2_3])
l14_3_0 = Feature_Extract_Visualization_Layer(None,(3,3),"SAME",(1,1),[l13])
l14_3_1 = Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l14_3_0])
#合并
l14 =  Feature_Extract_Visualization_Layer(None,(1,1),"SAME",(1,1),[l14_0,l14_1_2,l14_2_4,l14_3_1])

# imgs_path = "/home/night/data/VISUAL_NODE/I0_Train_imgs.npy"
# ori_imgs = np.load(imgs_path)
arr_path = "/home/night/data/key_points/"
# arr_to_layer_dict = {"I0_Conv2d_p1_3x3.npy": l1,
#                      "I0_Conv2d_p2_3x3.npy": l2,
#                      "I0_Conv2d_p3_3x3.npy": l3,
#                      "I0_MaxPool_p4_3x3.npy": l4,
#                      "I0_Conv2d_p5_1x1.npy": l5,
#                      "I0_Conv2d_p6_3x3.npy": l6,
#                      "I0_MaxPool_p7_3x3.npy": l7,
#                      "I0_Mixed_p8_a.npy": l8,
#                      "I0_Mixed_p8_b.npy": l9,
#                      "I0_Mixed_p8_c.npy": l10,
#                      }
arr_to_layer_dict ={"Conv2d_1a_3x3.npy":l1,
                     "Conv2d_2a_3x3.npy":l2,
                     "Conv2d_2b_3x3.npy":l3,
                     "MaxPool_3a_3x3.npy":l4,
                     "Conv2d_3b_1x1.npy":l5,
                     "Conv2d_4a_3x3.npy":l6,
                     "MaxPool_5a_3x3.npy":l7,
                     "Mixed_5b.npy":l8,
                     "Mixed_5c.npy":l9,
                     "Mixed_5d.npy":l10,
                     "Mixed_6a.npy":l11,
                     "Mixed_6b.npy":l12,
                     "Mixed_6c.npy":l13,
                     "Mixed_6d.npy":l14,
                     }
parent_dir = "visualization_graphs/"
if os.path.exists(parent_dir) == False:
    os.mkdir(parent_dir)
save_path_all = os.path.join(parent_dir, "features_heat_analyse")
if os.path.exists(save_path_all) == False:
    os.mkdir(save_path_all)

for arr_name in arr_to_layer_dict.keys():
    tmp_features = np.load(os.path.join(arr_path,arr_name))
    tmp_save_path = os.path.join(save_path_all, arr_name)
    if os.path.exists(tmp_save_path) == False:
        os.mkdir(tmp_save_path)
    # tmp_features = tmp_features[0]
    print tmp_features.shape
    for i in range(tmp_features.shape[3]):
        tmp_save_path_i = os.path.join(tmp_save_path, "{0}/".format(i))
        if os.path.exists(tmp_save_path_i) == False:
            os.mkdir(tmp_save_path_i)
        for j in range(5):
        # for j in range(tmp_features.shape[0]):
            print arr_name,i,j
            tmp_feature = tmp_features[j,:,:,i]
            tmp_feature = tmp_feature[:,:]
            sns.heatmap(tmp_feature, center=0)
            fig = plt.gcf()
            fig.savefig(os.path.join(tmp_save_path_i,"{0}.png".format(j)))
            plt.clf()

