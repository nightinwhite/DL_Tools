from Data_Argument import *
import tensorflow as tf
import cv2
import numpy as np
print tf.random_uniform([], maxval=4, dtype=tf.int32)
net_image = tf.placeholder(tf.float32,[480,640,3])
net_res = slice_and_resize (net_image)

real_image = cv2.imread("img_1.jpg")
real_image = real_image/255.- 0.5
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

res = sess.run(net_res, feed_dict={net_image:real_image})
print np.min(res),np.max(res)
res = (res - np.min(res))/(np.max(res) - np.min(res))*255.
res = np.asarray(res, np.uint8)
cv2.imshow("1",res)
cv2.waitKey(0)
cv2.destroyAllWindows()
