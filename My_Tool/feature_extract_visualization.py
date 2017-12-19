import numpy as np
class Feature_Extract_Visualization_Layer(object):
    def __init__(self, feature_value,kernel,padding,stride,prev_layers):
        self.prev_layers = prev_layers
        self.kernel = kernel
        self.feature_value = feature_value
        self.stride = stride
        if padding == "SAME":
            self.padding = ((kernel[0]-1)/2,(kernel[1]-1)/2)
        elif padding == "VALID":
            self.padding = (0,0)

    def get_box_from_point(self,x, y, kernel, pad, stride):
        # x_max y_max not include
        kernel_x = kernel[0]
        kernel_y = kernel[1]
        pad_x = pad[0]
        pad_y = pad[1]
        stride_x = stride[0]
        stride_y = stride[1]

        x_min = x * stride_x - pad_x
        x_max = x * stride_x - pad_x + kernel_x
        y_min = y * stride_y - pad_y
        y_max = y * stride_y - pad_y + kernel_y
        return x_min, y_min, x_max, y_max

    def map_to_ori_image(self,i,j):
        if self.prev_layers is None:
            return self.get_box_from_point(i,j,self.kernel,self.padding,self.stride)
        else:
            ori_box_res = []
            for prev_layer in self.prev_layers:
                x_min, y_min, x_max, y_max = self.get_box_from_point(i,j,self.kernel,self.padding,self.stride)
                ori_x_min1, ori_y_min1, _, _ = prev_layer.map_to_ori_image(x_min, y_min)
                _, ori_y_min2, ori_x_max2, _ = prev_layer.map_to_ori_image(x_max - 1, y_min)
                assert ori_y_min1 == ori_y_min2
                ori_y_min = ori_y_min1
                ori_x_min3, _, _, ori_y_max3 = prev_layer.map_to_ori_image(x_min, y_max - 1)
                assert ori_x_min1 == ori_x_min3
                ori_x_min = ori_x_min1
                _, _, ori_x_max4, ori_y_max4 = prev_layer.map_to_ori_image(x_max - 1, y_max - 1)
                assert ori_x_max2 == ori_x_max4
                ori_x_max = ori_x_max2
                assert ori_y_max3 == ori_y_max4
                ori_y_max = ori_y_max3
                ori_box_res.append([ori_x_min, ori_y_min ,ori_x_max, ori_y_max])
            ori_box_res = np.asarray(ori_box_res)
            x_min = np.min(ori_box_res[:,0])
            y_min = np.min(ori_box_res[:,1])
            x_max = np.max(ori_box_res[:,2])
            y_max = np.max(ori_box_res[:,3])
            return x_min,y_min,x_max,y_max




