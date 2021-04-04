'''
Author: wang shuli
Date: 2021-03-04 18:52:38
LastEditTime: 2021-04-03 00:42:49
LastEditors: your name
Description: 
'''
'''
Author: wang shuli
Date: 2021-03-04 18:52:38
LastEditTime: 2021-03-04 18:52:38
LastEditors: your name
Description: 
'''
import matplotlib.pyplot as plt

save_path = 'pics/'

#%%
k = 'Wetlands'
ora_score = 0.9735
oob_score = 0.9815
feature_score = \
[0.896484375, 0.970654296875, 0.9799967447916667, 0.9816487630208334, 0.9832763671875, 0.9840901692708334, 0.9841959635416667, 0.9846516927083333, 0.9853190104166667, 0.9855387369791667]
fig = plt.figure()
plt.plot(range(len(feature_score)), feature_score, 'g-', marker='^')
plt.hlines(ora_score, 0, len(feature_score)-1, linestyles='dashed', color='blue', label='ACC')
plt.hlines(oob_score, 0, len(feature_score)-1, linestyles='dashed', color='red', label='ACC1')
plt.legend()
plt.savefig(save_path+k+'_score.png', bbox_inches='tight')
plt.show()

#%%
Acc = [0.8718960979517106, 0.9454895678376627, 0.9420528664606935, 0.8765762673074724, 0.6656253575924019, 0.9679864210245259, 0.9624594728611207, 0.932948087119045]
Acc1 = [0.8452149368730213, 0.9453369950795286, 0.9422626540031278, 0.8895563947057253, 0.7138536064385704, 0.9649349658618455, 0.9580005340046535, 0.9469847808673763]
Acc2 = [0.9236525918297287, 0.9620818552847389, 0.9644925048632567, 0.9239348514322767, 0.8597780066369148, 0.9761414349467902, 0.9740969599877941, 0.9513064042415227]
fig = plt.figure()
plt.plot(Acc, 'g', marker='^', label='ACC')
plt.plot(Acc1, 'b', marker='*', label='ACC1')
plt.plot(Acc2, 'r', marker='o', label='ACC2')
# plt.hlines(ora_score, 0, len(feature_score)-1, linestyles='dashed', label='ACC')
# plt.hlines(oob_score, 0, len(feature_score)-1, linestyles='dashed', color='red', label='ACC1')
keys = ['barren', 'forest', 'grass-crops', 'Shrubland', 'Snow-Ice', 'Urban', 'Water', 'Wetlands']
plt.xticks(range(len(keys)), keys, rotation=45)
plt.ylim([0.6, 1])
plt.ylabel('Acc of cross validation')
plt.legend()
plt.savefig(save_path+'all_acc.png', bbox_inches='tight')
# plt.show()
# %%
from feature_selection import num2latex

keys = '荒漠、森林、草/作物、灌木、冰雪、城市、水、湿地'
keys = keys.split('、')
fs_list = []
with open('./log/tmp.txt', 'r') as f:
    for line in f:
        if line.startswith('feature set is'):
            fs = line.split('[')[-1][:-2]
            fs = list(map(int, fs.split(',')))
            fs_list.append(num2latex(fs))

def int2str(Acc):
    Acc = list(map(str, Acc))
    Acc = [a[:6] for a in Acc]
    return Acc

for itm in zip(keys, int2str(Acc), int2str(Acc1), int2str(Acc2), fs_list):
    print(' & '.join(itm) + '\\\\')

#%%
import matplotlib.pyplot as plt
import numpy as np 


imps = np.random.rand((100))
args_n = np.argsort(imps)[::-1]
imps_sort = np.sort(imps)[::-1]
fig = plt.figure()
plt.plot(range(len(imps_sort)), imps_sort, 'g--', marker='.', zorder=1)
# plt.scatter(range(len(imps_sort)), imps_sort, c='green', marker='.')
plt.ylabel('Variable importance')
args = [args_n[0], args_n[5], args_n[10]]

args_new = [np.where(args_n == a)[0][0] for a in args]
plt.scatter(args_new, imps_sort[args_new], c='red', marker='.', zorder=2)
plt.show()
# %%
