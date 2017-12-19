import numpy as np
from PIL import Image
class Weights_Visualization(object):
    def __init__(self,category, weight = None, pool_size = None, stride = None,prev = None):
        self.category = category
        self.weight = weight
        # self.weight = (self.weight - np.min(self.weight))/(np.max(self.weight) - np.min(self.weight))*1000.
        self.weight = np.clip(self.weight,0,None)
        self.pool_size = pool_size#(h,w)
        self.stride = stride
        self.prev = prev
        self.shape = None
        self.visual_map = None
        self.min_alpha = 150
        self._build()

    def _build(self):
        if self.category == "conv2d":
            if self.prev is None:
                self.visual_map = self.weight.transpose([3,0,1,2])
            else:
                prev_visual_map = self.prev.visual_map
                prev_stride = self.prev.stride
                visual_map_shape_h = (self.weight.shape[0] - 1) * prev_stride + 1 + prev_visual_map.shape[1] - 1
                visual_map_shape_w = (self.weight.shape[1] - 1) * prev_stride + 1 + prev_visual_map.shape[2] - 1
                tmp_h,tmp_w,tmp_in,tmp_out = self.weight.shape
                tmp_visual_map = []
                for j in range(tmp_out):
                    tmp_single_visual_map = np.full((visual_map_shape_h,visual_map_shape_w,3),0.)
                    for i in range(tmp_in):
                        for p in range(tmp_h):
                            for q in range(tmp_w):
                                h_b = prev_stride*p
                                h_e = prev_stride*p + prev_visual_map.shape[1]
                                w_b = prev_stride*q
                                w_e = prev_stride*q + prev_visual_map.shape[2]
                                #method 2
                                tmp_mask = tmp_single_visual_map[h_b:h_e, w_b:w_e,:] > prev_visual_map[i]*self.weight[p,q,i,j]
                                tmp_mask = np.asarray(tmp_mask,np.uint8)
                                tmp_single_visual_map[h_b:h_e, w_b:w_e, :] = tmp_single_visual_map[h_b:h_e, w_b:w_e, :]*tmp_mask + prev_visual_map[i]*self.weight[p,q,i,j]*(1-tmp_mask)
                                #method 1
                                # tmp_single_visual_map[h_b:h_e, w_b:w_e,:] += prev_visual_map[i]*self.weight[p,q,i,j]
                    tmp_visual_map.append(tmp_single_visual_map)
                self.visual_map = np.asarray(tmp_visual_map)
        elif self.category == "max_pool2d":
            prev_visual_map = self.prev.visual_map
            prev_stride = self.prev.stride
            visual_map_shape_h = (self.pool_size[0] - 1) * prev_stride + 1 + prev_visual_map.shape[1] - 1
            visual_map_shape_w = (self.pool_size[1] - 1) * prev_stride + 1 + prev_visual_map.shape[2] - 1
            tmp_prev_num = prev_visual_map.shape[0]
            tmp_visual_map = []
            for i in range(tmp_prev_num):
                tmp_single_visual_map = np.full((visual_map_shape_h, visual_map_shape_w, 3), 0.)
                h_b = 0
                h_e = prev_visual_map.shape[1]
                w_b = 0
                w_e = 0 + prev_visual_map.shape[2]
                tmp_single_visual_map[h_b:h_e, w_b:w_e, :] += prev_visual_map[i]
                tmp_visual_map.append(tmp_single_visual_map)
            self.visual_map = np.asarray(tmp_visual_map)
        elif self.category == "avg_pool2d":
            prev_visual_map = self.prev.visual_map
            prev_stride = self.prev.stride
            visual_map_shape_h = (self.pool_size[0] - 1) * prev_stride + 1 + prev_visual_map.shape[1] - 1
            visual_map_shape_w = (self.pool_size[1] - 1) * prev_stride + 1 + prev_visual_map.shape[2] - 1
            tmp_prev_num = prev_visual_map.shape[0]
            tmp_h = self.pool_size[0]
            tmp_w = self.pool_size[1]
            tmp_visual_map = []
            for i in range(tmp_prev_num):
                tmp_single_visual_map = np.full((visual_map_shape_h, visual_map_shape_w, 3), 0.)
                for p in range(tmp_h):
                    for q in range(tmp_w):
                        h_b = prev_stride * p
                        h_e = prev_stride * p + prev_visual_map.shape[1]
                        w_b = prev_stride * q
                        w_e = prev_stride * q + prev_visual_map.shape[2]
                        # method 2
                        tmp_mask = tmp_single_visual_map[h_b:h_e, w_b:w_e, :] > prev_visual_map[i]
                        tmp_mask = np.asarray(tmp_mask, np.uint8)
                        tmp_single_visual_map[h_b:h_e, w_b:w_e, :] = tmp_single_visual_map[h_b:h_e, w_b:w_e,:] * tmp_mask + prev_visual_map[i]* (1 - tmp_mask)

                tmp_visual_map.append(tmp_single_visual_map)
            self.visual_map = np.asarray(tmp_visual_map)

    def get_visual_images(self):
        res_images = []
        map_avgs = np.average(self.visual_map,(1,2,3))
        min_avg = np.min(map_avgs)
        max_avg = np.max(map_avgs)
        for i in range(self.visual_map.shape[0]):
            tmp_img_data = self.visual_map[i]
            tmp_avg = np.average(tmp_img_data)
            tmp_alpha = int((255. - self.min_alpha)*(tmp_avg - min_avg)/(max_avg - min_avg)) + self.min_alpha
            tmp_mask = np.full((tmp_img_data.shape[0],tmp_img_data.shape[1],1),tmp_alpha)
            tmp_max = np.max(tmp_img_data)
            tmp_min = np.min(tmp_img_data)
            if tmp_max == tmp_min:
                tmp_img_data = np.full_like(tmp_img_data,1) * 255.
            else:
                tmp_img_data = (tmp_img_data - tmp_min)/(tmp_max - tmp_min)*255.
            tmp_img_data = np.concatenate([tmp_img_data,tmp_mask],axis=2)
            tmp_img_data = np.asarray(tmp_img_data,np.uint8)
            tmp_img = Image.fromarray(tmp_img_data)
            res_images.append(tmp_img)
        return res_images

class Weights_Visualization_Combine_Node(object):
    def __init__(self,stride, prev_list):

        self.stride = stride
        self.prev_list = prev_list
        self.visual_map = None
        self.min_alpha = 150
        self._build()

    def _build(self):
        max_h = 0
        max_w = 0
        for prev in self.prev_list:
            if prev.visual_map.shape[1] > max_h:
                max_h = prev.visual_map.shape[1]
            if prev.visual_map.shape[2] > max_w:
                max_w = prev.visual_map.shape[2]
        tmp_res = []
        for prev in self.prev_list:
            pad_h = (max_h - prev.visual_map.shape[1])/2
            pad_w = (max_w - prev.visual_map.shape[2])/2
            # print prev.visual_map.shape, pad_h, pad_w
            tmp_res.append(np.pad(prev.visual_map,((0,0),(pad_h,pad_h),(pad_w, pad_w),(0,0)), "wrap"))
            # print prev.visual_map.shape
        self.visual_map = np.concatenate(tuple(tmp_res),axis=0)


    def get_visual_images(self):
        res_images = []
        map_avgs = np.average(self.visual_map,(1,2,3))
        min_avg = np.min(map_avgs)
        max_avg = np.max(map_avgs)
        for i in range(self.visual_map.shape[0]):
            tmp_img_data = self.visual_map[i]
            tmp_avg = np.average(tmp_img_data)
            tmp_alpha = int((255. - self.min_alpha)*(tmp_avg - min_avg)/(max_avg - min_avg)) + self.min_alpha
            tmp_mask = np.full((tmp_img_data.shape[0],tmp_img_data.shape[1],1),tmp_alpha)
            tmp_max = np.max(tmp_img_data)
            tmp_min = np.min(tmp_img_data)
            if tmp_max == tmp_min:
                tmp_img_data = np.full_like(tmp_img_data,1) * 255.
            else:
                tmp_img_data = (tmp_img_data - tmp_min)/(tmp_max - tmp_min)*255.
            tmp_img_data = np.concatenate([tmp_img_data,tmp_mask],axis=2)
            tmp_img_data = np.asarray(tmp_img_data,np.uint8)
            tmp_img = Image.fromarray(tmp_img_data)
            res_images.append(tmp_img)
        return res_images

