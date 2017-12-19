import plotly.graph_objs as plgo
from plotly.offline import plot
import numpy as np
import plotly.plotly as py

node_path = "/home/night/data/VISUAL_NODE/"
conv2d_p1 = np.load(node_path+"I0_Conv2d_p1_3x3.npy")
tst_map = conv2d_p1[0,0,::-1,:,3]
# tst_map = np.transpose(tst_map,[1,0])
trace = plgo.Heatmap(z = tst_map)
data = [trace]
fig = plgo.Figure(data = data)
py.image.save_as(figure_or_data = fig,filename="tmp_img.png")