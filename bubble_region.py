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
# cv2.imshow('orig', gray)

# ガウシアンフィルタ
gray = cv2.GaussianBlur(gray, (3, 3), 0)
# cv2.imshow('gaussian', gray)


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
# cv2.imshow('binarization', gray)

# Cannyフィルター
edges1 = cv2.Canny(gray, 150, 200)
cv2.imshow('edges', edges1)


# ラベリング処理(詳細版)
label = cv2.connectedComponentsWithStats(gray)

# オブジェクト情報を項目別に抽出
# data[0]にラベル１の情報が入っている
# nlabelsは背景0を含む。最大ラベルはnlabel-1
n = label[0] - 1
data = np.delete(label[2], 0, 0)
center = np.delete(label[3], 0, 0)  # 重心座標

# ラベリング結果書き出し用に二値画像をカラー変換
color_src = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# ラベリング結果書き出し用に二値画像をカラー変換
imgl = np.zeros((label[1].shape))

size = label[1].shape[:2]  # 画像サイズ
# print("size:", size)  # サイズの出力


# nlabel-1個のオブジェクトに応じた色を準備
cols = []
for i in range(0, n):  # cols[0],…,cols[nlabel-1]まで。
    cols.append(np.array([random.randint(0, 255),
                          random.randint(0, 255), random.randint(0, 255)]))

# print(cols)

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
# cv2.imshow('label', color_src)
# print(n)


# ラベルごとに解析する
for i in range(0, n):  # リストは0から。ラベルは1から。iは0から。
    if data[i][4] > 10 and data[i][4] < 1000:
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
blackImage = np.zeros((label[1].shape[0], label[1].shape[1], 3), np.uint8)

# print(blackImage.shape[:3])

# 真っ黒の画像をコピー
centerRegion3 = blackImage.copy()  # カラーでラベリングを表現
centerRegion2 = blackImage.copy()  # 矩形の中心点を描画
centerRegion1 = blackImage.copy()  # 重心情報を描画する
centerRegion0 = blackImage.copy()  # 中心の白領域のみの画像(加工前)
# cv2.imshow('black', blackImage)

# print(label[2])
# print(data)
# print(center)
countWhiteRegion = 0


# ラベリング情報を選別するためのリスト
delete_data = []


# ラベリングデータを厳選する
select_n = 0  # バブル中心の領域の個数
select_data = []  # バブル中心の領域のデータ
select_gravity = []  # バブル中心の重心座標
select_center = []  # バブル中心の領域の中心座標
select_cols = []  # ラベリングするために必要

# label[1]を見るためのループ
# for i in range(480):
#     print("{}".format(i) + "行目")
# print(label[1][i])

# ラベルごとに解析する2
for i in range(0, n):  # リストは0から。ラベルは1から。iは0から。
    if data[i][4] > 10 and data[i][4] < 1000:
        countWhiteRegion += 1
        select_n += 1
        # 色を塗る
        centerRegion1[label[1] == i+1, ] = [255, 255, 255]
        centerRegion0[label[1] == i+1, ] = [255, 255, 255]
        centerRegion2[label[1] == i+1, ] = [255, 255, 255]
        centerRegion3[label[1] == i+1, ] = cols[i]
        # centerRegion1[label[1] == i+1, ] = testCols[i]
        # 各オブジェクトの外接矩形を赤枠で表示
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        cv2.rectangle(centerRegion1, (x0, y0), (x1, y1), (0, 0, 255))
        cv2.rectangle(centerRegion2, (x0, y0), (x1, y1), (0, 0, 255))

        # 各オブジェクトのラベル番号と面積に黄文字で表示
        # cv2.putText(centerRegion1, "ID:" + str(i) + " S:" + str(data[i][4]) + " ("+str(
        #     data[i][2])+","+str(data[i][3])+")", (x0, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        #cv2.putText(centerRegion1, "S: " +str(data[i][4]), (x0, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

        # cv2.putText(centerRegion3, "label: " + str(
        #     i), (x1 - 10, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

        # 各オブジェクトの重心座標を黄文字で表示
        cv2.putText(centerRegion1, "XG: " + str(
            int(center[i][0])), (x1 - 10, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        cv2.putText(centerRegion1, "YG: " + str(
            int(center[i][1])), (x1 - 10, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

        select_data.append(data[i])
        select_gravity.append(np.array(center[i]))
        select_cols.append(np.array(cols[i]))

        # 矩形の中心
        cx = x0 + data[i][2]/2
        cy = y0 + data[i][3]/2
        rectangleCenter = [cx, cy]
        select_center.append(np.array(rectangleCenter))

        # 各オブジェクトの矩形の中心座標を紫文字で表示
        cv2.putText(centerRegion2, "CX: " + str(
            int(cx)), (x1 - 10, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255))
        cv2.putText(centerRegion2, "CY: " + str(
            int(cy)), (x1 - 10, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255))

    else:
        # color_src[label[1] == i+1, ] = [0, 0, 0]
        delete_data.append(i)


# print(delete_data)
# data = np.delete(label[2], 0, 0)
# for i in range(len(delete_data),0,-1):
#     select_data = np.delete(select_data, delete_data[i], 0)
#     select_gravity = np.delete(select_gravity, delete_data[i], 0)
#     select_n -= 1

# print("data[0]: " + str(type(data)))
# print("select_data[0]: " + str(type(select_data)))
# for i in range(select_n):
#     print(select_data[i])
# for i in range(select_n):
#     print(select_gravity[i])

# print(select_center)

# print(select_n)

print("白い領域は" + str(countWhiteRegion) + "個あります")
# print(center[0])
# Cannyフィルター(気泡中心の白い領域だけ抽出)
edges2 = cv2.Canny(centerRegion0, 100, 200)

contourImage = blackImage.copy()
contourImage = edges1 - edges2




# バブル外形の各点の座標を取得
# 関数を作る
outerPoint = []
outerPointNumber = 0
outerpointLabel = 0


for y in range(color_src.shape[0]):
    for x in range(color_src.shape[1]):
        if contourImage[y][x] == 255:
            outerPointNumber += 1
            # どれに近いか比較
            # 重心ver

            # L2距離を保持するリスト
            distanceLists = []

            # L2距離の計算
            for i in range(select_n):  # select_nはバブル中心の白領域の数
                # ラベリング時の順番でリストに値が追加されていく
                distanceX = select_gravity[i][0] - x
                distanceY = select_gravity[i][1] - y
                l2Distance = np.sqrt(distanceX ** 2 + distanceY ** 2)
                distanceLists.append(l2Distance)

            # print(distanceLists)
            # print(distanceLists.index(np.min(distanceLists)))
            outerpointLabel = distanceLists.index(np.min(distanceLists))
            # print(outerpointLabel)

            outerPoint.append(
                np.array([x, y, outerpointLabel]))

test = blackImage
outerLabeled = blackImage
print("[バブル外形各点の座標]")
print("[x, y, 番号, ラベル]")
for i in range(outerPointNumber):
    print("{}点目: ".format(i+1) + str(outerPoint[i]))
    # test[outerPoint[i][1]][outerPoint[i][0]] = [0, 255, 255]
    # バブル外形にラベリング

    outerLabeled[outerPoint[i][1]][outerPoint[i]
                                   [0]] = select_cols[outerPoint[i][2]]

labeCanny = centerRegion3 + outerLabeled

# print("shape[0]: " + str(color_src.shape[0]))
# print("shape[1]: " + str(color_src.shape[1]))

# cv2.imwrite("./saveImages/bubbleCanny.jpg", edges1)
# cv2.imwrite("./saveImages/bubbleInsideWhite.jpg", centerRegion1)
# cv2.imwrite("./saveImages/bubbleInsideCanny.jpg", edges2)
cv2.imwrite("./saveImages/labelCannyCenter.jpg", centerRegion3)
cv2.imwrite("./saveImages/labelCannyOuter.jpg", outerLabeled)
cv2.imwrite("./saveImages/labelCanny.jpg", labeCanny)

cv2.imshow('object', color_src)
cv2.imshow('edgesInside', edges2)

cv2.imshow('contourImage', contourImage)
cv2.imshow('outerLabeled', outerLabeled)  # バブル外形にラベリングしたもの
# cv2.imshow('test', test)

# cv2.imshow('centerRegion1', centerRegion1)
# cv2.imshow('centerRegion2', centerRegion2)
cv2.imshow('centerRegion3', centerRegion3)
cv2.imshow('labeCanny', labeCanny)

cv2.moveWindow('centerRegion3', 0, 80)
cv2.moveWindow('outerLabeled', 640, 80)
cv2.moveWindow('labeCanny', 0, 480)

cv2.waitKey(0)

cv2.destroyAllWindows()

# 画像の保存
#cv2.imwrite('sample_label1.jpg', color_src)
