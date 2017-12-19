import cv2
import numpy as np

from My_Tool.feature_extract_visualization import Feature_Extract_Visualization_Layer


def show_img(img,index):
    tmp_img = (img - np.min(img))/(np.max(img) - np.min(img))*255.
    tmp_img = np.stack((tmp_img[:,:,2],tmp_img[:,:,1],tmp_img[:,:,0]),axis=2)
    tmp_img = np.asarray(tmp_img,np.uint8)
    tmp_img = cv2.resize(tmp_img,(123,123),interpolation=cv2.INTER_NEAREST)
    cv2.imshow("{0}".format(index),tmp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def show_single_img(img,index):
    if np.max(img) != np.min(img):
        tmp_img = (img - np.min(img))/(np.max(img) - np.min(img))*255.
    else:
        tmp_img = np.full_like(img,0.)
    tmp_img = np.asarray(tmp_img,np.uint8)
    tmp_img = cv2.resize(tmp_img,(123,123),interpolation=cv2.INTER_NEAREST)
    cv2.imshow("{0}".format(index),tmp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
node_path = "/home/night/data/VISUAL_NODE/"
train_imgs = np.load(node_path+"I0_Train_imgs.npy")
val_imgs = np.load(node_path+"I0_val_images.npy")
conv2d_p1 = np.load(node_path+"I0_Conv2d_p1_3x3.npy")
conv2d_p2 = np.load(node_path+"I0_Conv2d_p2_3x3.npy")

print conv2d_p2.shape
l1 = Feature_Extract_Visualization_Layer(conv2d_p1[0],(3,3),"VALID",(2,2),None)
l2 = Feature_Extract_Visualization_Layer(conv2d_p2[0],(3,3),"SAME",(1,1),[l1])
for i in range(conv2d_p2.shape[-1]):
    map_index = i
    tst_map = conv2d_p2[0, 0, :, :, map_index]
    show_single_img(tst_map,0)
    print np.min(tst_map),np.max(tst_map),np.argmax(tst_map)
    tmp_i = np.argmax(tst_map) / tst_map.shape[1]
    tmp_j = np.argmax(tst_map) % tst_map.shape[1]
    print tmp_i,tmp_j
    x_min, y_min, x_max, y_max = l2.map_to_ori_image(tmp_i, tmp_j)
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max >= train_imgs.shape[1]:
        x_max = train_imgs.shape[1] - 1
    if y_max >= train_imgs.shape[2]:
        y_max = train_imgs.shape[2] - 1
    print x_min, y_min, x_max, y_max
    tst_img = train_imgs[0, x_min:x_max, y_min:y_max, :]
    show_img(tst_img, 0)
