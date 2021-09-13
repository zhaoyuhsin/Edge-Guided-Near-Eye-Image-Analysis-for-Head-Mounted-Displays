import cv2, math
import numpy as np
from skimage import draw
im = np.ones((240, 320, 3))
im = im.astype(np.uint8)
im = im * 255
print(im.dtype)
def roa(coor, a):
    x, y = coor
    xx = math.cos(a) * x - math.sin(a) * y
    yy = math.sin(a) * x + math.cos(a) * y
    return xx, yy
def calc_bbox(ellipse):
    x = ellipse[0]
    y = ellipse[1]
    a = ellipse[2]
    b = ellipse[3]
    alpha = ellipse[4]
    ans = []
    xx, yy = roa((x, y), -alpha)
    ans.append((xx - a, yy - b))
    ans.append((xx - a, yy + b))
    ans.append((xx + a, yy + b))
    ans.append((xx + a, yy - b))
    for i in range(len(ans)):
        ans[i] = roa(ans[i], alpha)
    return ans
def calc_bbox_iou(box1, box2, shape = (240, 320)):
    im = np.zeros(shape, dtype="uint8")
    im1 = np.zeros(shape, dtype="uint8")
    mask = cv2.fillPoly(im, np.expand_dims(box1, axis=0), 255)
    mask1 = cv2.fillPoly(im1, np.expand_dims(box2, axis=0), 255)
    mask_and = cv2.bitwise_and(mask, mask1)
    mask_or = cv2.bitwise_or(mask, mask1)
    iou = 1.0 * np.sum(mask_and) / np.sum(mask_or)
    #print('IoU : ', iou)
    return iou
def draw_circle(im, x, y, r):
    rr, cc = draw.circle(y, x, r)
    rr = rr.clip(6, im.shape[0] - 6)
    cc = cc.clip(6, im.shape[1] - 6)
    im[rr, cc, ...] = [0, 255, 0]
def draw_lines(ans):
    for i in range(len(ans)):
        x, y = ans[i]
        xx, yy = ans[(i +  1) % len(ans)]
        rr, cc = draw.line(int(y), int(x), int(yy), int(xx))
        rr = rr.clip(6, im.shape[0] - 6)
        cc = cc.clip(6, im.shape[1] - 6)
        im[rr, cc, ...] = [0, 255, 0]
def calc_ell_bbox_iou(ell1, ell2):
    ans1 = calc_bbox(ell1)
    ans2 = calc_bbox(ell2)
    return calc_bbox_iou(np.array(ans1, dtype = np.int32), np.array(ans2, dtype = np.int32))

if __name__ == '__main__':
    print('Hello Wordld......')
    exit(0)
    el_iris = [150, 100, 40, 60, 1.047]  # [264, 119, 58, 72, 0.15927]
    el_pupil = [150, 100, 20, 20, 0]  # 45 degree
    el_60 = [264, 119, 58, 72, 1.24719]  # 60degree
    ans = calc_bbox(el_iris)
    print(ans)
    print(np.array(ans))

    ans2 = calc_bbox(el_pupil)
    print('IoU : ', calc_bbox_iou(np.array(ans, dtype=np.int32), np.array(ans2, dtype=np.int32)))

    # for item in ans:
    #     x, y = item
    #     draw_circle(im, x, y, 2)
    draw_lines(ans)
    [rr_i, cc_i] = draw.ellipse_perimeter(int(el_iris[1]),
                                          int(el_iris[0]),
                                          int(el_iris[3]),
                                          int(el_iris[2]),
                                          orientation=el_iris[4])
    # [rr_p, cc_p] = draw.ellipse_perimeter(int(el_pupil[1]),
    #                                               int(el_pupil[0]),
    #                                               int(el_pupil[3]),
    #                                               int(el_pupil[2]),
    #                                               orientation=el_pupil[4])
    # [rr_60, cc_60] = draw.ellipse_perimeter(int(el_60[1]),
    #                                               int(el_60[0]),
    #                                               int(el_60[3]),
    #                                               int(el_60[2]),
    #                                               orientation=el_60[4])

    rr_i = rr_i.clip(6, im.shape[0] - 6)
    cc_i = cc_i.clip(6, im.shape[1] - 6)
    # rr_p = rr_p.clip(6, im.shape[0]-6)
    # cc_p = cc_p.clip(6, im.shape[1]-6)
    # rr_60 = rr_60.clip(6, im.shape[0]-6)
    # cc_60 = cc_60.clip(6, im.shape[1]-6)
    im[rr_i, cc_i, ...] = np.array([0, 0, 254])
    # im[rr_p, cc_p, ...] = np.array([255, 0, 0])
    # im[rr_60, cc_60, ...] = np.array([0, 255, 0])
    import matplotlib.pyplot as plt

    plt.imshow(im, plt.cm.gray)
    plt.pause(100.05)
