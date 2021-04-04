#%%
from sklearn.ensemble import RandomForestClassifier 
import numpy as np
from osgeo import gdal
from time_cut import find_common_name, crop_img_from_mask
from sklearn.model_selection import train_test_split, cross_val_score
from feature_selection import make_feature

#%%
def get_one_train_data(idx, x=None, y=None):
    tif_name, qa_name = tif_qa_name[idx]
    if x == None or y == None:
        x = np.random.randint(0, 4000)
        y = np.random.randint(0, 4000)
    print(x, y)
    tif, cloud = crop_img_from_mask(tif_name, qa_name, x, y, win_size)
    X = np.reshape(np.transpose(tif, (1,2,0)), (-1, tif.shape[0]))
    # r_b = np.clip(X[:, 3] / X[:, 0], 0.5, 2.5)
    # nir_b = np.clip(X[:, 4] / X[:, 0], 0.5, 2.5)
    # X = np.concatenate((X, r_b[:, None], nir_b[:, None]), 1)
    Y = cloud.flatten()
    return X, Y

def get_train_data(idx):
    X_train, y_train = [], [],
    for i in idx:
        print(i)
        X, Y = get_one_train_data(i)
        X_train.append(X)
        y_train.append(Y)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    return X_train, y_train

# %%
win_size = 500
tif_qa_name = find_common_name()
idx = np.random.randint(0, 118, 10)
clf = RandomForestClassifier(n_estimators=20, max_depth=5, min_samples_leaf=100, oob_score=True)

X_train, y_train = get_train_data(range(10, 20))
X_train = np.clip(X_train, 1, 8000)
clf.fit(X_train, y_train)
print(clf.oob_score_)
#%%
# , 1210, 3451
X_test, y_test = get_one_train_data(15)
print(clf.score(X_test, y_test))

#%%
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt 

img = np.reshape(X_test[:, 3:0:-1], (win_size, win_size, -1))
img = np.clip(img*5e-4, 0, 1)
segments = slic(img, n_segments=100, compactness=10)
out=mark_boundaries(img, segments)
plt.imshow(out)
plt.show()
# %%
import matplotlib.pyplot as plt 

img = np.reshape(X_test[:, 3:0:-1], (win_size, win_size, -1))
plt.imshow(img*5e-4)
plt.figure()
plt.imshow(y_test.reshape((win_size, win_size)))

y_pred = clf.predict_proba(X_test)
plt.figure()
plt.imshow(y_pred[:, 1].reshape((win_size, win_size)))
#%%
clf.fit(X_train, y_train)


# %%
