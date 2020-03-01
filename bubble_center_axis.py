# バブル内側の中心点を求める
# ラベリング後の画像は荒くなったのであまり使えない

import cv2
import numpy as np
import sys
import random

binary_src = cv2.imread("saveImages/bubbleInsideWhite.jpg", 0)

# print(binary_src.shape[:3])


# ラベリング処理(詳細版)
label = cv2.connectedComponentsWithStats(binary_src)

n = label[0] - 1
data = np.delete(label[2], 0, 0)
center = np.delete(label[3], 0, 0)
print(n)
print(data)
print(center)

# ラベリング結果書き出し用に二値画像をカラー変換
color_src = cv2.cvtColor(binary_src, cv2.COLOR_GRAY2BGR)

# nlabel-1個のオブジェクトに応じた色を準備
cols = []
for i in range(0, n):  # cols[0],…,cols[nlabel-1]まで。
    cols.append(np.array([random.randint(0, 255),
                          random.randint(0, 255), random.randint(0, 255)]))

for i in range(0, n):  # range(1,3)は1,2となる。3は含まれない。リストは0から始まる。
    color_src[label[1] == i+1, ] = cols[i]
cv2.imshow('label', color_src)


cv2.imshow("Binarization(Input)", binary_src)


cv2.waitKey(0)
cv2.destroyAllWindows()
