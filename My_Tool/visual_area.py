import numpy as np
def get_box_from_point(x,y,kernel,pad,stride):

    kernel_x = kernel[0]
    kernel_y = kernel[1]
    pad_x = pad[0]
    pad_y = pad[1]
    stride_x = stride[0]
    stride_y = stride[1]

    x_min = (x - 1) * stride_x + 1 - pad_x
    x_max = (x - 1) * stride_x - pad_x + kernel_x
    y_min = (y - 1) * stride_y + 1 - pad_y
    y_max = (y - 1) * stride_y - pad_y + kernel_y

    return x_min, y_min, x_max, y_max

def get_convd_size(W, H, kernel, pad, stride):
    kernel_x = kernel[0]
    kernel_y = kernel[1]
    pad_x = pad[0]
    pad_y = pad[1]
    stride_x = stride[0]
    stride_y = stride[1]
    H_res = int((H + 2 * pad_y - kernel_y)/stride_y) + 1
    W_res = int((W + 2 * pad_x - kernel_x)/stride_x) + 1
    return W_res, H_res

def get_original_size(W, H, kernel, pad, stride):
    kernel_x = kernel[0]
    kernel_y = kernel[1]
    pad_x = pad[0]
    pad_y = pad[1]
    stride_x = stride[0]
    stride_y = stride[1]
    H_res = (H - 1) * stride_y + kernel_y - 2 * pad_y
    W_res = (W - 1) * stride_x + kernel_x - 2 * pad_x
    return W_res, H_res

def single_map_value(value_rec, kernel, pad, stride):
    W = value_rec.shape[0]
    H = value_rec.shape[1]
    W_ori, H_ori = get_original_size(W, H, kernel, pad, stride)
    res_rec = np.full([W_ori, H_ori],0.)
    for i in range(W):
        for j in range(H):
            tmp_v = value_rec[i, j]
            x_min, y_min, x_max, y_max = get_box_from_point(i, j, kernel, pad, stride)
            give_v = (tmp_v+0.)/((x_max + 1 - x_min)*(y_max + 1 - y_min))
            for p in range(x_min, x_max+1):
                for q in range(y_min, y_max+1):
                    if p >= 0 and p < W_ori and q >=0 and q < H_ori:
                        res_rec[p, q] += give_v
    return res_rec

def multiple_map_value(value_rec, params_list):
    tmp_res = value_rec
    for params in params_list:
        kernel = params[0]
        pad = params[1]
        stride = params[2]
        tmp_res = single_map_value(tmp_res, kernel, pad, stride)
    return tmp_res

# tst_area = np.full([40,40],0.)
# tst_area[20,20] = 1.
# res_area = single_map_value(value_rec=tst_area, kernel=[5, 5], pad=[2,2], stride=[1,1])
# for i in range(res_area.shape[0]):
#     for j in range(res_area.shape[1]):
#         print i,j,res_area[i, j]


