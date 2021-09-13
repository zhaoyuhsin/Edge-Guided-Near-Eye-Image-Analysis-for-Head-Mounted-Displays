import pickle
import numpy as np
import pandas as pd

ds_list = ['LPW', 'NVG', 'Ope', 'Fuh']
method_list = ['', 'edge']
for ds in ds_list:
    ious = [0, 0]
    for i in range(2):
        filename = 'img/' + ds + '_' + method_list[i] + '_ious.pkl'
        f = open(filename, 'rb')
        ious[i] = pickle.load(f)
        f.close()
    iris_delta = ious[0][:,1] - ious[1][:,1]
    pupil_delta = ious[0][:,2] - ious[1][:,2]
    sort_pupil_index = np.argsort(pupil_delta)
    sort_iris_index = np.argsort(iris_delta)
    baseline_ious = ious[0]
    edge_ious = ious[1]
    # obtain iris index
    ans_id = []
    for id in sort_iris_index:
        if (edge_ious[id][1] < 0.8 or baseline_ious[id][1] < 0.2 or iris_delta[id] > -0.1): continue
        print(id, iris_delta[id], edge_ious[id][1], baseline_ious[id][1])
        ans_id.append(id)
        if (len(ans_id) > 100): break
    filename = 'img/index_file/' + ds + '_iris_index.pkl'
    print('!!!iris index -> ', filename, 'shape : ', len(ans_id))
    f = open(filename, 'wb')
    pickle.dump(ans_id, f)
    f.close()

    # obtain pupil index
    ans_id = []
    for id in sort_pupil_index:
        if (edge_ious[id][2] < 0.9 or baseline_ious[id][2] < 0.2 or pupil_delta[id] > -0.1): continue
        print(id, pupil_delta[id], edge_ious[id][2], baseline_ious[id][2])
        ans_id.append(id)
        if (len(ans_id) > 100): break
    filename = 'img/index_file/' + ds + '_pupil_index.pkl'
    print('!!!pupil index -> ', filename, 'shape : ', len(ans_id))
    f = open(filename, 'wb')
    pickle.dump(ans_id, f)
    f.close()

exit(0)
f = open('model_score/lpw_baseline.pkl', 'rb')
baseline_ious=pickle.load(f)
f.close()

f = open('model_score/lpw_edge.pkl', 'rb')
edge_ious=pickle.load(f)
f.close()

f = open('model_score/lpw_only_edge.pkl', 'rb')
only_edge_ious=pickle.load(f)
f.close()
print(edge_ious.shape, only_edge_ious.shape)

pupil_delta = baseline_ious[:,2]  - edge_ious[:, 2]
iris_delta  = baseline_ious[:, 1] - edge_ious[:, 1]
sort_pupil_index = np.argsort(pupil_delta)
sort_iris_index = np.argsort(iris_delta)

ans_id = []
for id in sort_pupil_index:
    if(edge_ious[id][2] < 0.2 or baseline_ious[id][2] < 0.2 or iris_delta[id] > -0.1):continue
    print(id, iris_delta[id], edge_ious[id][2], baseline_ious[id][2])
    ans_id.append(id)
    if(len(ans_id) > 100):break
f = open('model_score/pupil_index_new.pkl', 'wb')
pickle.dump(ans_id, f)
f.close()


exit(0)
ans_id = []
for id in sort_iris_index:
    if(edge_ious[id][1] < 0.4 or baseline_ious[id][1] < 0.4 or iris_delta[id] > -0.1):continue
    print(id, iris_delta[id], edge_ious[id][1], baseline_ious[id][1])
    ans_id.append(id)
    if(len(ans_id) > 100):break
f = open('model_score/iris_index_new.pkl', 'wb')
pickle.dump(ans_id, f)
f.close()

exit(0)
f = open('model_score/test.pkl', 'wb')
pickle.dump(a, f)
f.close()
exit(0)






