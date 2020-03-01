import cv2
import numpy as np
import sys
import random

# カーネル
kernel = np.ones((3, 3), np.uint8)

# 画像の読み込み
#img = cv2.imread('P1010441.jpg')
img = cv2.imread('87.jpg')
# グレースケール化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('orig', gray)

# ガウシアンフィルタ
gray = cv2.GaussianBlur(gray, (3, 3), 0)
# cv2.imshow('gaussian', gray)

# 大津の二値化
#ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
print(ret)

cv2.imshow("Src", img)
cv2.imshow("Otsu", binary)
cv2.waitKey(0)

# ラベリング処理(詳細版)
label = cv2.connectedComponentsWithStats(binary)

# オブジェクト情報を項目別に抽出
# data[0]にラベル１の情報が入っている
# nlabelsは背景0を含む。最大ラベルはnlabel-1
n = label[0] - 1
data = np.delete(label[2], 0, 0)
center = np.delete(label[3], 0, 0)  # 重心座標

print("{}個領域が検出されました".format(n))

# 真っ黒の画像を生成
blackImage = np.zeros((label[1].shape[0], label[1].shape[1], 3), np.uint8)

# ラベリング結果書き出し用に二値画像をカラー変換
labelingCenterWhite = blackImage

# nlabel-1個のオブジェクトに応じた色を準備
cols = []
for i in range(0, n):  # cols[0],…,cols[nlabel-1]まで。
    cols.append(np.array([random.randint(0, 255),
                          random.randint(0, 255), random.randint(0, 255)]))

# ラベルごとに解析する(バブル中心の白い領域を抽出)
for i in range(0, n):  # リストは0から。ラベルは1から。iは0から。
    if data[i][4] > 10 and data[i][4] < 1000:
        # 色を塗る
        labelingCenterWhite[label[1] == i+1, ] = cols[i]
        # labelingCenterWhite[label[1] == i+1, ] = testCols[i]
        # 各オブジェクトの外接矩形を赤枠で表示
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        # cv2.rectangle(labelingCenterWhite, (x0, y0), (x1, y1), (0, 0, 255))

        # 各オブジェクトのラベル番号と面積に黄文字で表示
        # cv2.putText(labelingCenterWhite, "ID:" + str(i) + " S:" + str(data[i][4]) + " ("+str(
        #     data[i][2])+","+str(data[i][3])+")", (x0, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        #cv2.putText(labelingCenterWhite, "S: " +str(data[i][4]), (x0, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

        # 各オブジェクトの重心座標を黄文字で表示
        #cv2.putText(labelingCenterWhite, "XG: " + str(int(center[i][0])), (x1 - 10, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        #cv2.putText(labelingCenterWhite, "YG: " + str(int(center[i][1])), (x1 - 10, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
    else:
        labelingCenterWhite[label[1] == i+1, ] = [0, 0, 0]

cv2.imshow("labeling", labelingCenterWhite)
cv2.waitKey(0)

cv2.deleteAllWindows()
