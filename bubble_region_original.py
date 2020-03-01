import cv2
import numpy as np
import sys
import random

# カーネル
kernel = np.ones((3, 3), np.uint8)

# 画像の読み込み
#img = cv2.imread('P1010441.jpg')
img = cv2.imread('87.jpg')
#img = cv2.imread('sample.jpg')

# グレースケール化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('orig', gray)

# ガウシアンフィルタ
gray = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imshow('gaussian', gray)


# モルフォロジー
#gray = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel,iterations = 10)
# cv2.imshow('black',gray)
#gray1 = cv2.dilate(gray,kernel,iterations = 10)
# cv2.imshow('dilate',gray1)
#gray1 = cv2.erode(gray1,kernel,iterations = 10)
# cv2.imshow('erode',gray1)
#gray = cv2.absdiff(gray, gray1)
# cv2.imshow('diff',gray)

# 大津の二値化
#ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
print(ret)
cv2.imshow('binarization', gray)

# Cannyフィルター
edges1 = cv2.Canny(gray, 100, 200)
cv2.imshow('edges', edges1)


# ラベリング処理(詳細版)
label = cv2.connectedComponentsWithStats(gray)

# オブジェクト情報を項目別に抽出
# data[0]にラベル１の情報が入っている
# nlabelsは背景0を含む。最大ラベルはnlabel-1
n = label[0] - 1
data = np.delete(label[2], 0, 0)
center = np.delete(label[3], 0, 0)

# ラベリング結果書き出し用に二値画像をカラー変換
color_src = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# ラベリング結果書き出し用に二値画像をカラー変換
imgl = np.zeros((label[1].shape))

size = label[1].shape[:2]  # 画像サイズ
print("size:", size)  # サイズの出力


# nlabel-1個のオブジェクトに応じた色を準備
cols = []
for i in range(0, n):  # cols[0],…,cols[nlabel-1]まで。
    cols.append(np.array([random.randint(0, 255),
                          random.randint(0, 255), random.randint(0, 255)]))

testCols = []
testCols.append(np.array([0, 0, 255]))  # 赤
testCols.append(np.array([255, 0, 0]))  # 青
testCols.append(np.array([0, 255, 0]))  # 緑
testCols.append(np.array([255, 0, 255]))  # 紫
testCols.append(np.array([0, 255, 255]))  # 黄
testCols.append(np.array([255, 255, 0]))  #
testCols.append(np.array([255, 255, 125]))  #
testCols.append(np.array([125, 0, 255]))  #
testCols.append(np.array([0, 125, 255]))  #
testCols.append(np.array([255, 125, 0]))  #
testCols.append(np.array([125, 255, 255]))  #
testCols.append(np.array([0, 0, 125]))  #
# print(testCols)

# 色を塗る
for i in range(0, n):  # range(1,3)は1,2となる。3は含まれない。リストは0から始まる。
    color_src[label[1] == i+1, ] = cols[i]
cv2.imshow('label', color_src)
print(n)


# ラベルごとに解析する
for i in range(0, n):  # リストは0から。ラベルは1から。iは0から。
    if data[i][4] > 10 & data[i][4] < 1000:
        # 色を塗る
        color_src[label[1] == i+1, ] = cols[i]
        # color_src[label[1] == i+1, ] = testCols[i]
        # 各オブジェクトの外接矩形を赤枠で表示
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        cv2.rectangle(color_src, (x0, y0), (x1, y1), (0, 0, 255))

        # 各オブジェクトのラベル番号と面積に黄文字で表示
        cv2.putText(color_src, "ID:" + str(i) + " S:" + str(data[i][4]) + " ("+str(
            data[i][2])+","+str(data[i][3])+")", (x0, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        #cv2.putText(color_src, "S: " +str(data[i][4]), (x0, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

        # 各オブジェクトの重心座標を黄文字で表示
        #cv2.putText(color_src, "XG: " + str(int(center[i][0])), (x1 - 10, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        #cv2.putText(color_src, "YG: " + str(int(center[i][1])), (x1 - 10, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
    else:
        color_src[label[1] == i+1, ] = [0, 0, 0]

# 真っ黒の画像を生成
blankImage = np.zeros((label[1].shape[0], label[1].shape[1], 3), np.uint8)

# 真っ黒の画像をコピー
contourImage = blankImage.copy()

# cv2.imshow('black', blankImage)

print(label[2])
print(data)

# ラベルごとに解析する
for i in range(0, n):  # リストは0から。ラベルは1から。iは0から。
    if data[i][4] > 10 and data[i][4] < 1000:
        # 色を塗る
        blankImage[label[1] == i+1, ] = [255, 255, 255]
        # color_src[label[1] == i+1, ] = testCols[i]
        # 各オブジェクトの外接矩形を赤枠で表示
        # x0 = data[i][0]
        # y0 = data[i][1]
        # x1 = data[i][0] + data[i][2]
        # y1 = data[i][1] + data[i][3]
        # cv2.rectangle(color_src, (x0, y0), (x1, y1), (0, 0, 255))

        # 各オブジェクトのラベル番号と面積に黄文字で表示
        # cv2.putText(color_src, "ID:" + str(i) + " S:" + str(data[i][4]) + " ("+str(
        #     data[i][2])+","+str(data[i][3])+")", (x0, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        #cv2.putText(color_src, "S: " +str(data[i][4]), (x0, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

        # 各オブジェクトの重心座標を黄文字で表示
        #cv2.putText(color_src, "XG: " + str(int(center[i][0])), (x1 - 10, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        #cv2.putText(color_src, "YG: " + str(int(center[i][1])), (x1 - 10, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
    # else:
    #     color_src[label[1] == i+1, ] = [0, 0, 0]

# Cannyフィルター(気泡中心の白い領域だけ抽出)
edges2 = cv2.Canny(blankImage, 100, 200)

cv2.imwrite("./saveImages/bubbleCanny.jpg", edges1)
cv2.imwrite("./saveImages/bubbleInsideWhite.jpg", blankImage)
cv2.imwrite("./saveImages/bubbleInsideCanny.jpg", edges2)

cv2.imshow('object', color_src)
cv2.imshow('black', blankImage)
cv2.imshow('edgesInside', edges2)

contourImage = edges1 - edges2
cv2.imshow('contourImage', contourImage)

cv2.waitKey(0)

cv2.destroyAllWindows()

# 画像の保存
#cv2.imwrite('sample_label1.jpg', color_src)
