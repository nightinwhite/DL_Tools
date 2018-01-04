#coding:utf-8
from My_Tool.feature_extract_visualization import *
import time

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
a = time.time()
print l12.map_to_ori_image(10,10)
print time.time() - a