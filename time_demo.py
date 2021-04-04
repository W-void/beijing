'''
Author: wang shuli
Date: 2021-04-04 14:23:18
LastEditTime: 2021-04-04 16:07:15
LastEditors: your name
Description: 
'''
#%%
import numpy as np
import matplotlib.pyplot as plt

save_path = 'pics_time/'

#%%
x1, y1 = (0, 60, 80), (1, 50, 300)
plt.figure()
plt.plot(x1[:-1], y1[:-1], 'g', marker='o', label='clear')
plt.plot(x1[1:], y1[1:], 'r', marker='o', label='cloud')
plt.legend()
plt.savefig(save_path+'demo1.png')
plt.show()
# %%
x2, y2 = (0, 55, 70, 80), (1, 40, 100, 300)
plt.figure()
plt.plot(x2[:2], y2[:2], 'g', marker='o', label='clear')
plt.plot(x2[1:-1], y2[1:-1], 'b', marker='o', label='mix')
plt.plot(x2[2:], y2[2:], 'r', marker='o', label='cloud')
plt.legend()
plt.savefig(save_path+'demo2.png')
plt.show()
# %%
x3, y3 = (0, 50, 52, 60, 62, 70, 71, 80), (1, 40, 55, 75, 90, 135, 150, 300)
plt.figure()
plt.plot(x3[:2], y3[:2], 'g', marker='o', label='clear')
plt.plot(x3[2:4], y3[2:4], marker='o', label='mix1')
plt.plot(x3[4:6], y3[4:6], 'b', marker='o', label='mix2')
plt.plot(x3[-2:], y3[-2:], 'r', marker='o', label='cloud')
plt.legend()
plt.savefig(save_path+'demo3.png', bbox_inches='tight')
plt.show()
# %%
