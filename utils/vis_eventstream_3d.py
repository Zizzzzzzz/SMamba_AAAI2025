from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import cv2

events = np.array(np.load("D:/exp/240708/val_day_004_td/1890.npy", allow_pickle=True))
x = events[0]
y = events[1]
t = events[2]
p = events[3]

# Creating figure
fig = plt.figure()
ax = plt.axes(projection ="3d")
 
# Add x, y gridlines 
ax.grid(b = False, color ='white', 
		linestyle ='-.', linewidth = 0.3, 
		alpha = 0.2) 
cm=mpl.colors.ListedColormap(['b','r'])
# ax.scatter3D(t, x, 639-y, s=0.5, c=p,cmap=cm)
ax.scatter3D(t, x, 719-y, s=0.5, c=p,cmap=cm)
# plt.ylabel('', rotation=38)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.axis('off')
 
# plt.title("simple 3D scatter plot")
ax.set_xlabel('T-axis', fontweight ='bold') 
ax.set_ylabel('X-axis', fontweight ='bold') 
ax.set_zlabel('Y-axis', fontweight ='bold')
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
plt.savefig('events_3dnew.png', c = 'c')
plt.show()

