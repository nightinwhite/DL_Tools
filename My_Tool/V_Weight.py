#coding:utf-8
import numpy as np
from PIL import Image
from pprint import pprint
class V_Weight(object):
    def __init__(self,category, weight = None, pool_size = None, stride = None,prev = None):
        self.category = category
        self.weight = weight
        self.weight = np.clip(self.weight,0,None)
        self.pool_size = pool_size#(h,w)
        self.stride = stride
        self.prev = prev
        self.shape = None
        self.min_alpha = 100.
        self.max_alpha = 200.
        self.decay_rate = 0.5#透明淡化处理最小值
        self.test_k = 0
        self.visual_map = []
        self._build()

    def _build(self):
        if self.category == "conv2d":
            weight_shape = self.weight.shape
            if self.prev is None:
                weight_average_list = np.average(self.weight,(0,1,2))
                # print weight_average_list
                alpha_rate = (self.max_alpha - self.min_alpha) / (np.max(weight_average_list) - np.min(weight_average_list))#统计所有决定透明度
                alpha_bias = np.min(weight_average_list)
                for i in range(weight_shape[3]):
                    tmp_map_weight = self.weight[:,:,:,i]
                    tmp_shape = tmp_map_weight.shape
                    if np.max(tmp_map_weight) - np.min(self.weight) != 0:
                        tmp_color_rate = 255./(np.max(tmp_map_weight) - np.min(self.weight))#统计单个map决定颜色分量
                    else:
                        tmp_color_rate = 0
                    tmp_map_weight = tmp_map_weight * tmp_color_rate
                    tmp_map_weight = np.asarray(tmp_map_weight,np.uint8)
                    tmp_visual_map = np.full((tmp_shape[0],tmp_shape[1],4),0,np.uint8)
                    tmp_visual_map[:,:,:3] = tmp_map_weight
                    alpha_mask = (tmp_map_weight != (0,0,0)).any()
                    alpha_mask = np.asarray(alpha_mask,np.uint8)
                    tmp_alpha = (np.average(self.weight[:,:,:,i])-alpha_bias)*alpha_rate
                    tmp_alpha = alpha_mask * tmp_alpha + self.min_alpha
                    tmp_visual_map[:,:,3] = tmp_alpha
                    tmp_visual_map = Image.fromarray(tmp_visual_map)
                    self.visual_map.append(tmp_visual_map)
            else:
                weight_average_list = np.average(self.weight, (0, 1, 2))
                alpha_variance = np.max(weight_average_list) - np.min(weight_average_list)  # 统计所有决定透明度
                alpha_bias = np.min(weight_average_list)
                for i in range(weight_shape[3]):
                    tmp_map_weight = self.weight[:,:,:,i]
                    tmp_variance = np.max(tmp_map_weight) - np.min(tmp_map_weight)
                    tmp_bias = np.min(tmp_map_weight)
                    prev_visual_map_w,prev_visual_map_h = self.prev.visual_map[0].size
                    tmp_shape_h = tmp_map_weight.shape[0]*self.stride + (prev_visual_map_h-1) - (self.stride - 1)
                    tmp_shape_w = tmp_map_weight.shape[1]*self.stride + (prev_visual_map_w-1) - (self.stride - 1)
                    tmp_visual_map = np.full([tmp_shape_h,tmp_shape_w,4],0,np.uint8)
                    for j in range(weight_shape[2]):
                        tmp_map_weight = self.weight[:,:,j,i]
                        for p in range(weight_shape[0]):
                            for q in range(weight_shape[1]):
                                tmp_prev_map = self.prev.visual_map[j]
                                tmp_prev_map = np.asarray(tmp_prev_map)
                                tmp_prev_map.flags.writeable = True
                                tmp_mask = tmp_prev_map[:,:,3]
                                # print tmp_mask[0,0]
                                tmp_mask = tmp_mask * ((tmp_map_weight[p,q] - tmp_bias)/tmp_variance*(1 - self.decay_rate) + self.decay_rate)
                                tmp_mask = np.asarray(tmp_mask,np.uint8)
                                # print "!",j,tmp_map_weight[p,q],tmp_mask[0,0]
                                tmp_prev_map[:,:,3] = tmp_mask
                                # if p == 1 and q == 1:
                                #     print p,q,tmp_prev_map
                                #     print "!",tmp_visual_map[self.stride*p:self.stride*p+tmp_prev_map.shape[0],self.stride*q:self.stride*q+tmp_prev_map.shape[1],]
                                tmp_visual_map = self._paste(tmp_prev_map,tmp_visual_map,self.stride*p,self.stride*q)
                                # if p == 1 and q == 1:
                                #     print "!!",tmp_visual_map[self.stride*p:self.stride*p+tmp_prev_map.shape[0],self.stride*q:self.stride*q+tmp_prev_map.shape[1],]
                                # print tmp_visual_map
                                # tmp_prev_map = Image.fromarray(tmp_prev_map)
                                # tmp_visual_map.paste(tmp_prev_map,(self.stride*p,self.stride*q),tmp_prev_map)
                                # tst_img = Image.fromarray(tmp_visual_map)
                                # tst_img.save("test/{0}.png".format(self.test_k))
                                # self.test_k+=1
                    tmp_visual_map.flags.writeable = True
                    if self.weight.shape[3] != 1:
                        tmp_alpha = (np.average(self.weight[:, :, :, i]) - alpha_bias) / alpha_variance * (
                        1 - self.decay_rate) + self.decay_rate
                        tmp_visual_map_mask = tmp_visual_map[:, :, 3]
                        tmp_visual_map_mask = np.asarray(tmp_visual_map_mask, np.float)
                        tmp_visual_map_mask.flags.writeable = True
                        tmp_visual_map_mask *= tmp_alpha
                        tmp_visual_map[:, :, 3] = np.asarray(tmp_visual_map_mask, np.uint8)
                    self.visual_map.append(tmp_visual_map)
                if len(self.visual_map) != 1:
                    self.visual_map = np.asarray(self.visual_map, np.float)
                    visual_map_mask = self.visual_map[:, :, :, 3]
                    # print visual_map_mask
                    visual_map_mask = (visual_map_mask - np.min(visual_map_mask)) / (
                    np.max(visual_map_mask) - np.min(visual_map_mask)) * 255.
                    self.visual_map[:, :, :, 3] = visual_map_mask
                    self.visual_map = np.asarray(self.visual_map, np.uint8)
                tmp_list = []
                for map in self.visual_map:
                    tmp_list.append(Image.fromarray(map))
                self.visual_map = tmp_list

        elif self.category == "max_pool2d":
            assert self.prev is not None
            prev_visual_map = self.prev.visual_map
            for i in range(len(prev_visual_map)):
                prev_visual_map_w, prev_visual_map_h = self.prev.visual_map[0].size
                tmp_shape_h = self.pool_size[0] * self.stride + prev_visual_map_h-1 - (self.stride - 1)
                tmp_shape_w = self.pool_size[1] * self.stride + prev_visual_map_w-1 - (self.stride - 1)
                tmp_visual_map = np.full([tmp_shape_h, tmp_shape_w, 4], 0, np.uint8)
                tmp_visual_map = Image.fromarray(tmp_visual_map)
                tmp_prev_map = prev_visual_map[i]
                tmp_visual_map.paste(tmp_prev_map,(0,0),tmp_prev_map)
                self.visual_map.append(tmp_visual_map)

        elif self.category == "avg_pool2d":
            assert self.prev is not None
            prev_visual_map = self.prev.visual_map
            for i in range(len(prev_visual_map)):
                prev_visual_map_w, prev_visual_map_h = self.prev.visual_map[0].size
                tmp_shape_h = self.pool_size[0] * self.stride + prev_visual_map_h-1 - (self.stride - 1)
                tmp_shape_w = self.pool_size[1] * self.stride + prev_visual_map_w-1 - (self.stride - 1)
                tmp_visual_map = np.full([tmp_shape_h, tmp_shape_w, 4], 0, np.uint8)
                tmp_visual_map = Image.fromarray(tmp_visual_map)
                tmp_prev_map = prev_visual_map[i]
                for p in range(self.pool_size[0]):
                    for q in range(self.pool_size[1]):
                        tmp_visual_map.paste(tmp_prev_map, (self.stride * p, self.stride * q), tmp_prev_map)
                self.visual_map.append(tmp_visual_map)

    def _paste(self,a,b,pos_i,pos_j):
        return self._paste_type2(a,b,pos_i,pos_j)

    def _paste_type1(self,a,b,pos_i,pos_j):
        tmp_paste_aera = b[pos_i:pos_i + a.shape[0], pos_j:pos_j + a.shape[1]]
        a = np.asarray(a, np.int32)
        tmp_paste_aera = np.asarray(tmp_paste_aera, np.int32)
        tmp_dis = a - tmp_paste_aera
        tmp_mask = tmp_dis > 0
        tmp_mask = np.asarray(tmp_mask, np.uint8)
        tmp_dis = tmp_dis * tmp_mask
        tmp_dis = np.asarray(tmp_dis, np.uint8)
        b[pos_i:pos_i + a.shape[0], pos_j:pos_j + a.shape[1]] += tmp_dis
        return b

    def _paste_type2(self,a,b,pos_i,pos_j):
        # print a
        # print b[pos_i:pos_i + a.shape[0], pos_j:pos_j + a.shape[1]]
        imgb = Image.fromarray(b)
        imga = Image.fromarray(a)
        imgb.paste(imga,(pos_j,pos_i),imga)
        b = np.asarray(imgb)
        return b





