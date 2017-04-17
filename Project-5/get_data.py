import os
import numpy as np
import params
import pickle

from lesson_functions import extract_features
from fnmatch import fnmatch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

root = './dataset/'
pattern = "*.png"
show = True

data = []
labels = []

for i, dir in enumerate(['non-vehicles', 'vehicles']):
    for path, subdirs, files in os.walk(root + dir):
        for j in range(0, len(files), 10):
            name = files[j]
            if fnmatch(name, pattern):
                data.append(os.path.join(path, name))
                labels.append(i)
                labels.append(i)

data = np.array(extract_features(data, color_space=params.color_space, spatial_size=params.spatial_size,
                     hist_bins=params.hist_bins, orient=params.orient,
                     pix_per_cell=params.pix_per_cell, cell_per_block=params.cell_per_block, hog_channel=params.hog_channel,
                     spatial_feat=params.spatial_feat, hist_feat=params.hist_feat, hog_feat=params.hog_feat, show=show))
labels = np.array(labels)

norm = StandardScaler().fit(data)
data = norm.transform(data)

plt.plot(data[0])
plt.title("Feature normalization")
plt.show()

print("Data size:", data.shape)

index = pickle.load(open('best_index.p', 'rb'))

np.save("data", data)
np.save("labels", labels)
pickle.dump(norm, open('norm.p', 'wb'))