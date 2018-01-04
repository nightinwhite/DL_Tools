import os
path = "/home/night/data/key_points/"
name_list = os.listdir(path)
def compare_name(x,y):
    x_len = len(x)
    y_len = len(y)
    k_len = x_len
    if x_len > y_len:
        k_len = y_len
    for k in range(k_len):
        if ord(x[k]) > ord(y[k]):
            return 1
        elif ord(x[k]) < ord(y[k]):
            return -1
    if x_len > y_len:
        return 1
    else:
        return -1
name_list = sorted(name_list,cmp=compare_name)
for n in name_list:
    print "\"{0}\",".format(n.split(".")[0])