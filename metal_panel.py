import os
import cv2
import copy
import numpy as np

metal_panel = "C:\image_processing\metal_panel.jpg"

#画像入力
img = cv2.imread(metal_panel, 1)

#表示用関数
def imageshow(window_name, img_data):
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(window_name, img_data)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

#画像水平方向結合
def imgmerge(img1, img2):
    img_h = cv2.hconcat([img1, img2])
    return img_h

#グレースケール化、2値化、背景ノイズ(大)削除
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
denoise_img = cv2.fastNlMeansDenoising(gray_img, 10, 10, 7, 21)

ret, th1 = cv2.threshold(denoise_img, 90, 255, cv2.THRESH_BINARY)

#大領域のマスク作成
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations=1)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=4)
opening[:, -1] = 255
contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = list(filter(lambda x: cv2.contourArea(x) > 100000, contours))
mask_1 = np.zeros_like(closing)
cv2.drawContours(mask_1, contours, -1, color=255, thickness=-1)
mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, kernel, iterations=7)

#imageshow("mask1", mask_1)

#大領域マスク適用
contour_aply1 = copy.deepcopy(img)
contour_aply1[mask_1 == 0] = [0, 0, 0]
imageshow("mask1", contour_aply1)

#小領域のマスク作成
tcontour_img = copy.deepcopy(contour_aply1)
tcontour_img = cv2.cvtColor(tcontour_img, cv2.COLOR_BGR2GRAY)
ret, masked_th = cv2.threshold(tcontour_img, 100, 255, cv2.THRESH_BINARY)
#masked_th = masked_th.astype(np.uint8)

topening = cv2.morphologyEx(masked_th, cv2.MORPH_OPEN, kernel, iterations=6)
tclosing = cv2.morphologyEx(topening, cv2.MORPH_CLOSE, kernel, iterations=4)

ret, inv_img = cv2.threshold(tclosing, 100, 255, cv2.THRESH_BINARY_INV)

tcontours, thierarchy = cv2.findContours(inv_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
tcontours1 = list(filter(lambda x: cv2.contourArea(x) > 100000, tcontours))
tcontours1 = list(filter(lambda x: cv2.contourArea(x) < 5000000, tcontours1))

tcontours2 = list(filter(lambda x: cv2.contourArea(x) > 200, tcontours))
tcontours2 = list(filter(lambda x: cv2.contourArea(x) < 30000, tcontours2))
tcontours2 = list(filter(lambda x: (cv2.moments(x)["m01"] / cv2.moments(x)["m00"]) > 1500, tcontours2))
mask_2 = np.zeros_like(tclosing)
cv2.drawContours(mask_2, tcontours1, -1, color=255, thickness=-1)
cv2.drawContours(mask_2, tcontours2, -1, color=255, thickness=-1)

#マスク適用
contour_aply1[mask_2 == 255] = [0, 0, 0]

imageshow("mask2", contour_aply1)