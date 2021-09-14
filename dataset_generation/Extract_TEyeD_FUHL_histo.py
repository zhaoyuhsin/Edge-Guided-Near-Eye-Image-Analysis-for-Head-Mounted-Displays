import os
import cv2
import sys
import glob
import copy
import argparse
import matplotlib
import numpy as np
import deepdish as dd
import scipy.io as scio

sys.path.append('.')

from PIL import Image
from sklearn.cluster import KMeans
from matplotlib.patches import Circle, Ellipse

from helperfunctions import generateEmptyStorage, getValidPoints
from helperfunctions import ransac, ElliFit, my_ellipse, Circle_Fit

import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--noDisp', help='Specify flag to display labelled images', type=int, default=1)
parser.add_argument('--path2ds',
                    help='Path to dataset',
                    type=str,
                    default='../../Datasets')
args = parser.parse_args()

if args.noDisp:
    noDisp = True
    print('No graphics')
else:
    noDisp = False
    print('Showing figures')


gui_env = ['Qt5Agg','WXAgg','TKAgg','GTKAgg']
for gui in gui_env:
    try:
        print("testing: {}".format(gui))
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

def readFormattedText(path2file, ignoreLines):
    data = []
    count = 0
    f = open(path2file, 'r')
    for line in f:
        if count > ignoreLines:
            d = [float(d) for d in line.split(';') if d is not '\n']
            data.append(d)
        count = count + 1

    f.close()
    return data


print("Using: {}".format(matplotlib.get_backend()))
import matplotlib.pyplot as plt
plt.ion()

PATH_DIR = os.path.join(args.path2ds, 'FUHL/DIKABLISVIDEOS/')
PATH_LABEL = os.path.join(args.path2ds, 'FUHL//DIKABLISANNOTATIONS/ANNOTATIONS/')
PATH_DS = os.path.join(args.path2ds, 'Histogram')
PATH_MASTER = os.path.join(args.path2ds, 'Histogram_mat')

# curDir, dirs, files in os.walk("test")
list_ds = list(os.walk(PATH_DIR))[0][2] #'DikablisT_9_4.mp4', 'DikablisT_12_2.mp4', 'DikablisT_25_12.mp4',
list_ds.sort()

print(len(list_ds))
# Remove bad videos. We cant unzip them. 11 videos.
bad_item = ['DikablisT_18_3.mp4', 'DikablisT_18_4.mp4', 'DikablisT_18_5.mp4', 'DikablisT_18_6.mp4', 'DikablisT_18_7.mp4', 'DikablisT_18_8.mp4', 'DikablisT_18_9.mp4', 'DikablisT_19_1.mp4', 'DikablisT_28_4.mp4', 'DikablisT_29_1.mp4', 'DikablisT_3_2.mp4']
for item in bad_item:
    print(item)
    list_ds.remove(item)
print(len(list_ds))
print('Extracting TEyeD-FUHL')
sc = (640.0/384.0)

Image_counter = 0.0
ds_num = 0

comming = 0

pic_num = 11200 # 2500
fix_interval = 5000000 // pic_num # 5665053
ds_name = 'Fuhl_{}'.format(pic_num)
# Generate empty dictionaries
Data, keydict = generateEmptyStorage(name='Fuhl', subset='Fuhl_{}'.format(pic_num))

for name in list_ds:
    # print('!!!!!!!: ', name)
    # print('!!!!!: ', os.path.join(PATH_DIR, name))
    opts = glob.glob(os.path.join(PATH_DIR, name))

    for Path2vid in opts:
        Name = Path2vid.split('/')[-1].split('.')[-2]

        path_iris_param = os.path.join(PATH_LABEL, "{}.mp4iris_eli.txt".format(Name))
        path_pupil_param = os.path.join(PATH_LABEL, "{}.mp4pupil_eli.txt".format(Name))
        path_eye_ball_param = os.path.join(PATH_LABEL, "{}.mp4eye_ball.txt".format(Name))
        path_lid = os.path.join(PATH_LABEL, "{}.mp4lid_lm_2D.txt".format(Name))

        iris_param = np.array(readFormattedText(path_iris_param, 0))
        pupil_param = np.array(readFormattedText(path_pupil_param, 0))
        eye_ball_param = np.array(readFormattedText(path_eye_ball_param, 0))
        eye_lid_param = np.array(readFormattedText(path_lid, 0))

        VidObj = cv2.VideoCapture(Path2vid)
        
        if not noDisp:
            fig, plts = plt.subplots(1, 1)
        fr_num = 0
            
        while (VidObj.isOpened()):
            ret, I = VidObj.read()
            if ret == True and I is not None:
                # if I.shape
                iris_list = iris_param[fr_num, :]
                pupil_list = pupil_param[fr_num, :]
                eye_ball_list = eye_ball_param[fr_num, :]
                eye_lid_list = eye_lid_param[fr_num, :]
            
                fr_num += 1

                if len(keydict['archive']) == pic_num:
                    print('!!!!! num: ', pic_num)
                    break
                comming += 1
                if comming % fix_interval != 0:
                    continue
                if eye_ball_list[2] < 0 or eye_ball_list[3] < 0 or eye_ball_list[1] < 0:
                    continue
                if iris_list[2] < 0 or iris_list[3] < 0:
                    continue
                if pupil_list[2] < 0 or pupil_list[3] < 0:
                    continue
                
                # Get the eyelid
                eye_lid = []
                for i in range(2, 35, 2):
                    eye_lid.append([int(float(eye_lid_list[i]) * sc), int(float(eye_lid_list[i + 1])* sc)])
                for i in range(68, 35, -2):
                    eye_lid.append([int(float(eye_lid_list[i]) * sc), int(float(eye_lid_list[i + 1])* sc)])
                
                eye_lid = np.array(eye_lid)

                I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                I = cv2.resize(I, (640, 480), cv2.INTER_LANCZOS4)

                # Draw the mask image, channel 1, value 0 1 2 3
                maskIm_woskin = np.zeros([I.shape[0], I.shape[1]], np.int8)

                eye_ball_list[1:4] = eye_ball_list[1:4] * sc
                iris_list[2:6] = iris_list[2:6] * sc
                pupil_list[2:6] = pupil_list[2:6] * sc

                cv2.circle(maskIm_woskin, (int(eye_ball_list[2]), int(eye_ball_list[3])), int(eye_ball_list[1]), 1, -1)
                cv2.ellipse(maskIm_woskin, (int(iris_list[2]), int(iris_list[3])),
                            (int(iris_list[4] / 2), int(iris_list[5] / 2)),
                            iris_list[1], 0, 360, 2, -1)
                cv2.ellipse(maskIm_woskin, (int(pupil_list[2]), int(pupil_list[3])),
                            (int(pupil_list[4] / 2), int(pupil_list[5] / 2)),
                            pupil_list[1], 0, 360, 3, -1)

                # Draw the mask with skin
                maskIm_inskin = maskIm_woskin.copy()
                tmp = np.zeros([I.shape[0], I.shape[1]], np.int8)
                cv2.fillPoly(tmp, [eye_lid], 1)
                maskIm_inskin[tmp == 0] = 0

                pupil_loc = pupil_list[2:4]
                pupil_list[4:6] = pupil_list[4:6] / 2
                iris_list[4:6] = iris_list[4:6] / 2

                # plt.imshow(I)
                # plt.show()
                # plt.pause(1)
                # print('shape: ', I.shape)
                Data['Images'].append(I)
                Data['Masks'].append(maskIm_inskin)
                Data['Masks_noSkin'].append(maskIm_woskin)
                Data['Info'].append(str(comming))
                Data['pupil_loc'].append(pupil_loc)

                keydict['resolution'].append(I.shape)
                keydict['archive'].append(ds_name)
                keydict['pupil_loc'].append(pupil_loc)

                # print(np.append(pupil_list[2:6], pupil_list[1]).tolist(), np.append(iris_list[2:6], pupil_list[1]).tolist())
                if pupil_list[1] > 90:
                    pupil_list[1] = -(180 - pupil_list[1])
                if iris_list[1] > 90:
                    iris_list[1] = -(180 - iris_list[1])
                pupil_list[1] = np.deg2rad(pupil_list[1])
                iris_list[1] = np.deg2rad(iris_list[1])
                # print(np.append(pupil_list[2:6], pupil_list[1]).tolist(),
                #       np.append(iris_list[2:6], pupil_list[1]).tolist())
                Data['Fits']['pupil'].append(np.append(pupil_list[2:6], pupil_list[1]).tolist())
                Data['Fits']['iris'].append(np.append(iris_list[2:6], iris_list[1]).tolist())
                Data['Fits']['ball'].append(np.append(np.append(np.append(eye_ball_list[2:4], eye_ball_list[1]), eye_ball_list[1]), 0).tolist())

                
                print("{}, id_{}".format(Name, fr_num))
                if not noDisp:
                    if comming == 1:

                        cE = Ellipse(tuple(pupil_list[2:4]),
                                     2 * pupil_list[4],
                                     2 * pupil_list[5],
                                     angle=np.rad2deg(pupil_list[1]))
                        cL = Ellipse(tuple(iris_list[2:4]),
                                     2 * iris_list[4],
                                     2 * iris_list[5],
                                     angle=np.rad2deg(iris_list[1]))
                        cB = Circle(tuple(eye_ball_list[2:4]),
                                     eye_ball_list[1])

                        cE.set_facecolor('None')
                        cE.set_edgecolor((1.0, 0.0, 0.0))
                        cL.set_facecolor('None')
                        cL.set_edgecolor((0.0, 1.0, 0.0))
                        cB.set_facecolor('None')
                        cB.set_edgecolor((0.0, 0.0, 1.0))

                        cI = plts.imshow(I)
                        cM = plts.imshow(maskIm_woskin, alpha=0.5)
                        cX = plts.scatter(pupil_loc[0], pupil_loc[1])
                        plts.add_patch(cE)
                        plts.add_patch(cL)
                        plts.add_patch(cB)

                        plt.show()
                        plt.pause(0.01)
                    else:
                        cE.center = tuple(pupil_loc)
                        cE.angle = np.rad2deg(pupil_list[1])
                        cE.width = 2 * pupil_list[4]
                        cE.height = 2 * pupil_list[5]

                        cL.center = tuple(iris_list[2:4])
                        cL.width = 2 * iris_list[4]
                        cL.height = 2 * iris_list[5]
                        cL.angle = np.rad2deg(iris_list[1])

                        cB.center = tuple(eye_ball_list[2:4])
                        cB.radius = eye_ball_list[1]


                        newLoc = np.array([pupil_loc[0], pupil_loc[1]])
                        cI.set_data(I)
                        cM.set_data(maskIm_woskin)
                        cX.set_offsets(newLoc)
                        mypause(0.01)

                    # print('keydict[resolution] = ', keydict['resolution'])
            else:
                break
        print('num: ', len(keydict['resolution']))
 
keydict['resolution'] = np.stack(keydict['resolution'], axis=0)
keydict['archive'] = np.stack(keydict['archive'], axis=0)
keydict['pupil_loc'] = np.stack(keydict['pupil_loc'], axis=0)
Data['pupil_loc'] = np.stack(Data['pupil_loc'], axis=0)
Data['Images'] = np.stack(Data['Images'], axis=0)
Data['Masks'] = np.stack(Data['Masks'], axis=0)
Data['Masks_noSkin'] = np.stack(Data['Masks_noSkin'], axis=0)
Data['Fits']['pupil'] = np.stack(Data['Fits']['pupil'], axis=0)
Data['Fits']['iris'] = np.stack(Data['Fits']['iris'], axis=0)
Data['Fits']['ball'] = np.stack(Data['Fits']['ball'], axis=0)

print("Data['Fits']['pupil']", Data['Fits']['pupil'].shape)
print("Data['Fits']['iris']", Data['Fits']['iris'].shape)
print("Data['Fits']['ball']", Data['Fits']['ball'].shape)

warnings.filterwarnings("ignore")
# Save data
dd.io.save(os.path.join(PATH_DS, str(ds_name)+'.h5'), Data)
scio.savemat(os.path.join(PATH_MASTER, str(ds_name)+'.mat'), keydict, appendmat=True)

ds_num += 1