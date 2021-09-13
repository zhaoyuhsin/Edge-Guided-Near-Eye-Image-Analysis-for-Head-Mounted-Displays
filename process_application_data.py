import pickle
import numpy as np
import pandas as pd


mp = {}
t = {}
left_prob = {}
right_prob = {}
gt = []
# left eye
a = pd.read_csv('our_data_test/pupil0.csv', dtype = np.float64, header=None)
print('.... : ', a.shape, '\n', a.head())
for i in range(a.shape[0]):
    mp[a[9][i]] = i + 1
    t[i + 1] = a[9][i]
    left_prob[i + 1] = a[0][i]

# right eye
a = pd.read_csv('our_data_test/pupil1.csv', dtype = np.float64, header=None)
print('.... : ', a.shape, '\n', a.head())
for i in range(a.shape[0]):
    right_prob[i + 1] = a[0][i]

# gt
a = pd.read_csv('our_data_test/gt.csv', dtype = np.float64, header=None)
for i in range(a.shape[0]):
    gt.append(a[4][i])
print('.... : ', a.shape, '\n', a.head(), '\n', len(gt))

# dump (mp, left_prob, right_prob, gt)
filename = 'our_data_test/data.pkl'
#print('pickle.dump : {} itmes..'.format(len(app_center[0])))
f = open(filename, 'wb')
pickle.dump((mp, t, left_prob, right_prob, gt), f)
f.close()

exit(0)