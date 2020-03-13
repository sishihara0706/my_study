import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import random

# 標準出力の色を変えるために定義


class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RETURN = '\033[07m'  # 反転
    ACCENT = '\033[01m'  # 強調
    FLASH = '\033[05m'  # 点滅
    RED_FLASH = '\033[05;41m'  # 赤背景+点滅
    END = '\033[0m'


# カーネル
kernel = np.ones((3, 3), np.uint8)

# 画像の読み込み
arg = sys.argv
print(len(arg))
if len(arg) < 2:
  filename = "images/bubble-1.jpg"
elif len(arg) >= 2:
  number = arg[1]
  filename = "images/bubble-{}.jpg".format(number)
  inputLabel = int(arg[2])



# filename = "images/bubble-1.jpg"
img = cv2.imread(filename)
cv2.imshow('src', img)

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

# cv2.imshow("binary", binary)


# カーネル
kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], np.uint8)

# オープニング処理
# binary = cv2.erode(binary, kernel, iterations=3)
# binary = cv2.dilate(binary, kernel, iterations=3)


blackImage = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

reverseBinary = blackImage.copy()
# print(reverseBinary.shape[:3])

# 二値画像を反転
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        if binary[y][x] == 0:
            reverseBinary[y][x] = 255
        else:
            reverseBinary[y][x] = 0

# cv2.imshow("reverseBinary", reverseBinary)
# cv2.waitKey(0)

# cv2.imshow("Src", img)
# cv2.imshow("Otsu", binary)


# ラベリング処理(詳細版)

# バブルのラベリング
label = cv2.connectedComponentsWithStats(reverseBinary)

# バブル中心白い領域のラベリング
label_2 = cv2.connectedComponentsWithStats(binary)


# オブジェクト情報を項目別に抽出
# data[0]にラベル１の情報が入っている
# nlabelsは背景0を含む。最大ラベルはnlabel-1
n = label[0] - 1
data = np.delete(label[2], 0, 0)
center = np.delete(label[3], 0, 0)  # 重心座標

n_2 = label_2[0] - 1
labelNumber_2 = label_2[1]
data_2 = np.delete(label_2[2], 0, 0)
center_2 = np.delete(label_2[3], 0, 0)  # 重心座標


print("{}個領域が検出されました".format(n))

# print(data)


# 真っ黒の画像を生成
blackImage = np.zeros((label[1].shape[0], label[1].shape[1], 3), np.uint8)


labelingBubble = blackImage.copy()
labelingBubblePainted = blackImage.copy()

# 中心の白い領域を塗りつぶす
paintCenterWhiteRegion = reverseBinary.copy()

detectionWhiteRegion = blackImage.copy()
detectionWhiteRegion3 = blackImage.copy()

# ラベリングデータを厳選する
select_n = 0  # バブル中心の領域の個数
select_data = []  # バブル中心の領域のデータ
select_gravity = []  # バブル中心の重心座標
select_center = []  # バブル中心の領域の中心座標
select_cols = []  # ラベリングするために必要

# nlabel-1個のオブジェクトに応じた色を準備
cols = []
for i in range(0, n+1):  # cols[0],…,cols[nlabel-1]まで。
    cols.append(np.array([random.randint(0, 255),
                          random.randint(0, 255), random.randint(0, 255)]))

# 中心が白い領域のラベルを保存するリスト
saveCenterWhiteLabels = []

# binaryのラベリング
for i in range(0, n_2):  # リストは0から。ラベルは1から。iは0から。
    if data_2[i][4] > 10 and data_2[i][4] < 1500:
        select_n += 1
        # data_2はlabel[2]からlabel[2][0]を切り取ったものを代入しているためずれる
        paintCenterWhiteRegion[label_2[1] == i+1, ] = 255
        detectionWhiteRegion[label_2[1] == i+1, ] = 255
        detectionWhiteRegion3[label_2[1] == i+1, ] = cols[i+1]
        saveCenterWhiteLabels.append(i+1)

        x0 = data_2[i][0]
        y0 = data_2[i][1]
        x1 = data_2[i][0] + data_2[i][2]
        y1 = data_2[i][1] + data_2[i][3]

        select_data.append(np.array([data_2[i], i+1]))
        # ラベルも一緒に保存
        select_gravity.append(np.array(center_2[i] + [i+1]))
        select_cols.append(np.array([cols[i], i+1]))

        # 矩形の中心
        cx = x0 + data_2[i][2]/2
        cy = y0 + data_2[i][3]/2
        rectangleCenter = [cx, cy]
        select_center.append(np.array(rectangleCenter))

print("中心の白領域ラベル: {}".format(saveCenterWhiteLabels))
# print(select_data)
# print(select_data[1][1])
# バブル中心を塗りつぶしたラベリング
label_3 = cv2.connectedComponentsWithStats(paintCenterWhiteRegion)

n_3 = label_3[0] - 1
data_3 = np.delete(label_3[2], 0, 0)
center_3 = np.delete(label_3[3], 0, 0)  # 重心座標


print(n, n_3)  # ラベルの数は等しくなった
print(select_n)

# 等しい面積のラベルを保存するリスト
saveSameAreaLabel = []

# 異なる面積のラベルを保存するリスト
saveDifferentAreaLabel = []

bubbleHasWhiteRegion2 = blackImage.copy()


# ラベルごとに解析する(前処理)
# 中心に白い領域を持たない気泡(ぼやけた気泡)はカットする
for i in range(0, n):  # リストは0から。ラベルは1から。iは0から。
    if data[i][4] > 10 and data[i][4] < 17000:
        # 色を塗る
        labelingBubble[label[1] == i+1, ] = cols[i]
        labelingBubblePainted[label_3[1] == i+1, ] = cols[i]
        # labelingBubble[label[1] == i+1, ] = testCols[i]
        # 各オブジェクトの外接矩形を赤枠で表示
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]

        # 面積で比較
        # 面積が異なるラベルを保存
        if data[i][4] != data_3[i][4]:
            saveDifferentAreaLabel.append(i)
            # 面積が異なる(中心に白い領域を持つバブル)ラベルだけ色を塗る
            bubbleHasWhiteRegion2[label[1] == i+1, ] = cols[i]

        # 面積が等しいラベルを保存
        if data[i][4] == data_3[i][4]:
            saveSameAreaLabel.append(i)
        # cv2.rectangle(labelingBubble, (x0, y0), (x1, y1), (0, 0, 255))

        # 各オブジェクトのラベル番号と面積に黄文字で表示
        # cv2.putText(labelingBubble, "ID:" + str(i) + " S:" + str(data[i][4]) + " ("+str(
        #     data[i][2])+","+str(data[i][3])+")", (x0, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        # cv2.putText(labelingBubble, "S: " +str(data[i][4]), (x0, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

        # 各オブジェクトの重心座標を黄文字で表示
        #cv2.putText(labelingBubble, "XG: " + str(int(center[i][0])), (x1 - 10, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        #cv2.putText(labelingBubble, "YG: " + str(int(center[i][1])), (x1 - 10, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
    else:
        labelingBubble[label[1] == i+1, ] = [0, 0, 0]


bubbleHasWhiteRegion = blackImage.copy()

# 中心に白い領域を持つバブルだけ白く色を塗る
for i in range(len(saveDifferentAreaLabel)):
    bubbleHasWhiteRegion[label[1] == saveDifferentAreaLabel[i]+1, ] = 255

# 前処理を行った画像と中心の白い領域を取り出した画像を足し合わせる
result = bubbleHasWhiteRegion + detectionWhiteRegion

# チャンネルを1に変換
result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)


# ヒストグラム表示
# hist = cv2.calcHist([image], [0], None, [256], [0,256])
# plt.plot(hist)

# 輪郭検出
# imageはラベリング画像
# contoursは輪郭画素をオブジェクトごとに保持している
# hierarchyはオブジェクトの階層構造を保持している
image, contours, hierarchy = cv2.findContours(
    result, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


# cv2.imshow("labelingBubble", labelingBubble)
# cv2.imshow("labelingBubblePainted", labelingBubblePainted)
# cv2.imshow("paintCenterWhiteRegion", paintCenterWhiteRegion)
# cv2.imshow("bubbleHasWhiteRegion", bubbleHasWhiteRegion)
# 色付き
# cv2.imshow("bubbleHasWhiteRegion2", bubbleHasWhiteRegion2)
# cv2.imshow("result", result)
# cv2.imshow("detectionWhiteRegion", detectionWhiteRegion)


print(contours[0][0][0])

countourImg = blackImage.copy()

# 輪郭画像を生成
for i in range(len(contours)):
    cv2.polylines(countourImg, contours[i], True, (255, 255, 255), 1)

# 3チャンネルに変換(色を塗るため)
countourImg = cv2.cvtColor(countourImg, cv2.COLOR_BGR2GRAY)

# 白い中心領域と外形を対応づける
# バブル外形の各点の座標を取得
# 関数を作る
outerPoint = []
outerPointNumber = 0
outerpointLabel = 0

print("重心")
print(select_gravity)
#
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        if countourImg[y][x] == 255:
            # バブル外形が何点あるかカウントする
            outerPointNumber += 1

            # どれに近いか比較
            # 重心で比較

            # L2距離を保持するリスト
            distanceLists = []

            # L2距離の計算
            # バブル中心の白領域の数だけループを回す
            for i in range(select_n):  # select_nはバブル中心の白領域の数
                # ラベリング時の順番でリストに値が追加されていく
                distanceX = select_gravity[i][0] - x
                distanceY = select_gravity[i][1] - y
                l2Distance = np.sqrt(distanceX ** 2 + distanceY ** 2)
                distanceLists.append(l2Distance)

            # print(distanceLists)
            # print(distanceLists.index(np.min(distanceLists)))

            # 一番近い白い領域のインデックス番号(ラベル番号)を保存
            # outerpointLabel = distanceLists.index(np.min(distanceLists))
            outerpointLabel = saveCenterWhiteLabels[distanceLists.index(
                np.min(distanceLists))]
            # print(outerpointLabel)

            # [x座標, y座標, ラベル番号]の形で追加していく
            outerPoint.append(
                np.array([x, y, outerpointLabel]))


outerLabeled = blackImage.copy()

countLabel2 = 0
countLabel3 = 0
countLabel4 = 0
countLabel5 = 0
countLabel6 = 0
countLabel7 = 0
countLabel8 = 0
countLabel9 = 0
countLabel11 = 0
countLabel12 = 0
# print("[バブル外形各点の座標]")
# print("[x, y, 番号, ラベル]")
for i in range(outerPointNumber):
    print("{}点目: ".format(i+1) + str(outerPoint[i]))
    if outerPoint[i][2] == 2:
        countLabel2 += 1
    if outerPoint[i][2] == 3:
        countLabel3 += 1
    if outerPoint[i][2] == 4:
        countLabel4 += 1
    if outerPoint[i][2] == 5:
        countLabel5 += 1
    if outerPoint[i][2] == 6:
        countLabel6 += 1
    if outerPoint[i][2] == 7:
        countLabel7 += 1
    if outerPoint[i][2] == 8:
        countLabel8 += 1
    if outerPoint[i][2] == 9:
        countLabel9 += 1
    if outerPoint[i][2] == 11:
        countLabel11 += 1
    if outerPoint[i][2] == 12:
        countLabel12 += 1

    # outerLabeled[outerPoint[i][1]][outerPoint[i][0]] = select_cols[outerPoint[i][2]][0]
    outerLabeled[outerPoint[i][1]][outerPoint[i][0]] = cols[outerPoint[i][2]]

# print("バブル外形のラベル2の点数は{}個".format(countLabel2))
# print("バブル外形のラベル3の点数は{}個".format(countLabel3))
# print("バブル外形のラベル4の点数は{}個".format(countLabel4))
# print("バブル外形のラベル5の点数は{}個".format(countLabel5))
# print("バブル外形のラベル6の点数は{}個".format(countLabel6))
# print("バブル外形のラベル7の点数は{}個".format(countLabel7))
# print("バブル外形のラベル8の点数は{}個".format(countLabel8))
# print("バブル外形のラベル9の点数は{}個".format(countLabel9))
# print("バブル外形のラベル11の点数は{}個".format(countLabel11))
# print("バブル外形のラベル12の点数は{}個".format(countLabel12))

labelNumber_2Img = blackImage.copy()
# print(labelNumber_2[0])

# for y in range(img.shape[0]):
#   for x in range(img.shape[1]):
#     if labelNumber_2[y][x] == 2:
#       labelNumber_2Img[y][x] == [0,0,255]


# 中心の白い領域の画素数をカウント
# saveCenterWhiteLabelsは前景と面積10~1000のもの以外を引いた個数
# ラベル番号とラベルの個数を対応付けるためのリスト
whiteCentertCoordinate = []
countLabel = [0] * len(saveCenterWhiteLabels)
for i in range(len(saveCenterWhiteLabels)):
    for y in range(480):
        for x in range(640):
            # (labelNumber_2 = label[1])
            if labelNumber_2[y][x] == saveCenterWhiteLabels[i]:
                countLabel[i] += 1
                whiteCentertCoordinate.append(
                    np.array([x, y, saveCenterWhiteLabels[i]]))


whiteCenterLabelData = []

for i in range(len(saveCenterWhiteLabels)):
    # ラベル番号とラベルの個数を対応付ける
    whiteCenterLabelData.append([saveCenterWhiteLabels[i], countLabel[i]])
    print("ラベル{0}には画素が{1}個あります".format(
        saveCenterWhiteLabels[i], countLabel[i]))
    # print(whiteCenterLabelData[i])

# 累積をカウントするリスト
allAccumulation = []


# for i in range(len(whiteCentertCoordinate)):
#   for j in range(len(outerPoint)):
#     if whiteCentertCoordinate[i][2] == outerPoint[j][2]:
#       xc = whiteCentertCoordinate[i][0]
#       yc = whiteCentertCoordinate[i][1]
#       xo = outerPoint[j][0]
#       yo = outerPoint[j][1]
#       r = (xc - xo) ** 2 + (yc - yo) ** 2
#       r = int(np.sqrt(r))
#       accumulation[r] += 1

# 中心領域と外形のデータをラベルごとに分割する
# whiteCentertCoordinate, outerPoint


whiteCentertCoordinateByLabel = {}
for i in range(len(saveCenterWhiteLabels)):
    whiteCentertCoordinateByLabel[saveCenterWhiteLabels[i]] = []
    for j in range(len(whiteCentertCoordinate)):
        if whiteCentertCoordinate[j][2] == saveCenterWhiteLabels[i]:
            # 配列のラベル番目に格納する
            whiteCentertCoordinateByLabel[saveCenterWhiteLabels[i]].append(
                whiteCentertCoordinate[j])

# print("ラベル4の中心領域の画素データ")
# print(whiteCentertCoordinateByLabel[4])
# print(pycolor.RED+"ラベル4の中心領域の画素データは{}個です".format(len(whiteCentertCoordinateByLabel[4]))+pycolor.END)

outerPointByLabel = {}
for i in range(len(saveCenterWhiteLabels)):
    outerPointByLabel[saveCenterWhiteLabels[i]] = []
    for j in range(len(outerPoint)):
        if outerPoint[j][2] == saveCenterWhiteLabels[i]:
            # 配列のラベル番目に格納する
            outerPointByLabel[saveCenterWhiteLabels[i]].append(outerPoint[j])

print("ラベル4の外形の画素データ")
# print(outerPointByLabel[4])
print(pycolor.RED +
      "ラベル4の外形の画素データは{}個です".format(len(outerPointByLabel[4]))+pycolor.END)


# 一点ずつ比較する
accumulations = {}  # 累積を保存する accumulation[ラベル番号]でアクセス
accumulationByOnePoints = {}
for i in range(len(saveCenterWhiteLabels)):
    accumulations[saveCenterWhiteLabels[i]] = []
    accumulation = [0] * 150
    accumulationByOnePoints[saveCenterWhiteLabels[i]] = []
    for j in range(len(whiteCentertCoordinateByLabel[saveCenterWhiteLabels[i]])):
        accumulationByOnePoint = [0] * 150
        for k in range(len(outerPointByLabel[saveCenterWhiteLabels[i]])):
            xc = whiteCentertCoordinateByLabel[saveCenterWhiteLabels[i]][j][0]
            yc = whiteCentertCoordinateByLabel[saveCenterWhiteLabels[i]][j][1]
            xo = outerPointByLabel[saveCenterWhiteLabels[i]][k][0]
            yo = outerPointByLabel[saveCenterWhiteLabels[i]][k][1]
            r = (xc - xo) ** 2 + (yc - yo) ** 2
            r = int(np.sqrt(r))
            accumulation[r] += 1
            accumulationByOnePoint[r] += 1

        accumulationByOnePoints[saveCenterWhiteLabels[i]].append(
            np.array([accumulationByOnePoint, xc, yc]))

    accumulations[saveCenterWhiteLabels[i]].append(accumulation)


for i in range(len(saveCenterWhiteLabels)):
    print(pycolor.GREEN+"ラベル{}:".format(saveCenterWhiteLabels[i])+pycolor.END + "{}".format(
        accumulations[saveCenterWhiteLabels[i]][0]))
    print(pycolor.YELLOW+"ラベル{}の最大値: ".format(saveCenterWhiteLabels[i])+pycolor.END + "{}".format(
        max(accumulations[saveCenterWhiteLabels[i]][0])))
    print(pycolor.PURPLE+"ラベル{}の最大値が入っているインデックス番号(半径r): ".format(saveCenterWhiteLabels[i])+pycolor.END + "{}".format(
        accumulations[saveCenterWhiteLabels[i]][0].index(np.max(accumulations[saveCenterWhiteLabels[i]][0]))))


for i in range(len(saveCenterWhiteLabels)):
    print("ラベル{0}の累積の総和{1}".format(saveCenterWhiteLabels[i], sum(
        accumulations[saveCenterWhiteLabels[i]][0])))
# print(sum(accumulations[2]))


args = sys.argv


# 中心であろうデータを生成

circleData = {}

# 累積が最大になる点(円の中心と思われる点)を選ぶ処理
for i in range(len(saveCenterWhiteLabels)):
    circleData[saveCenterWhiteLabels[i]] = [0, 0, 0, 0]
    print("[ラベル{}]".format(saveCenterWhiteLabels[i]))
    for j in range(len(accumulationByOnePoints[saveCenterWhiteLabels[i]])):
        x = accumulationByOnePoints[saveCenterWhiteLabels[i]][j][1]
        y = accumulationByOnePoints[saveCenterWhiteLabels[i]][j][2]
        maxValue = np.max(
            accumulationByOnePoints[saveCenterWhiteLabels[i]][j][0])
        r = accumulationByOnePoints[saveCenterWhiteLabels[i]][j][0].index(
            np.max(accumulationByOnePoints[saveCenterWhiteLabels[i]][j][0]))

        print("Center({},{}), max={}, r={}".format(x, y, maxValue, r))
        # 累積が一番大きいものを選ぶ
        if circleData[saveCenterWhiteLabels[i]][2] < maxValue:
            circleData[saveCenterWhiteLabels[i]] = [x, y, maxValue, r]

    print("最大の累積", circleData[saveCenterWhiteLabels[i]])
    print("\n")

# print("[最大累積のデータ※]")
# for i in range(len(saveCenterWhiteLabels)):
#   print(pycolor.YELLOW+"ラベル{}: ".format(saveCenterWhiteLabels[i])+pycolor.END +  "max={}, ".format(max(accumulations[saveCenterWhiteLabels[i]][0]))+ "r={}".format(accumulations[saveCenterWhiteLabels[i]][0].index(np.max(accumulations[saveCenterWhiteLabels[i]][0]))))

print("[最大累積のデータ]")
for i in range(len(saveCenterWhiteLabels)):
    print("ラベル{0}: Center({1},{2}), max={3}, r={4}".format(saveCenterWhiteLabels[i], circleData[saveCenterWhiteLabels[i]][
          0], circleData[saveCenterWhiteLabels[i]][1], circleData[saveCenterWhiteLabels[i]][2], circleData[saveCenterWhiteLabels[i]][3]))


# ラベル番号をコマンドライン引数で与えた場合の処理
# print("コマンドライン入力: {}".format(args[1]))

# inputLabel = int(args[1])
# # print(accumulationByOnePoints[inputLabel])
# # print(accumulationByOnePoints[inputLabel][0])
# # print(accumulationByOnePoints[inputLabel][0][0])

# print("[ラベル{}]".format(inputLabel))
# for i in range(len(accumulationByOnePoints[inputLabel])):
#   print("Center({},{}), max={}, r={}".format(accumulationByOnePoints[inputLabel][i][1], accumulationByOnePoints[inputLabel][i][2], np.max(accumulationByOnePoints[inputLabel][i][0]), accumulationByOnePoints[inputLabel][i][0].index(np.max(accumulationByOnePoints[inputLabel][i][0]))))


# 各ラベルで最大累積データに該当するものを円として描画
for i in range(len(saveCenterWhiteLabels)):
    xc = circleData[saveCenterWhiteLabels[i]][0]
    yc = circleData[saveCenterWhiteLabels[i]][1]
    radius = circleData[saveCenterWhiteLabels[i]][3]
    cv2.circle(img, (xc, yc), radius, (0, 0, 255),
               thickness=2, lineType=cv2.LINE_8, shift=0)
    # 各オブジェクトの重心座標を黄文字で表示
    cv2.putText(img, "C:(" + str(xc) + "," + str(yc) + ")" + ", " + "r:" + str(radius), (xc - radius, yc + radius + 15),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 2)


# グラフの表示


for i in range(len(accumulationByOnePoints[inputLabel])):
  left = np.array(range(0,40))
  height = accumulationByOnePoints[inputLabel][i][0][0:40]
  plt.bar(left, height)
  plt.title("Center({},{}), max={}, r={}".format(accumulationByOnePoints[inputLabel][i][1], accumulationByOnePoints[inputLabel][i][2], np.max(accumulationByOnePoints[inputLabel][i][0]), accumulationByOnePoints[inputLabel][i][0].index(np.max(accumulationByOnePoints[inputLabel][i][0]))))
  plt.show()


# args = sys.argv
# print("コマンドライン入力: {}".format(args[1]))
# left = np.array(range(0,40))
# height = accumulations[int(args[1])][0][0:40]


# fig, ax = plt.subplots()
# x 軸 (major) の目盛りを設定する。
# ax.set_xticks(np.linspace(0, 1, ))


# plt.bar(left, height)
# plt.show()


# accumulation = [0] * 500
# for i in range(whiteCenterLabelData[0][1]):
#   for j in range(len(outerPoint)):
#     if whiteCentertCoordinate[i][2] == outerPoint[j][2]:
#       xc = whiteCentertCoordinate[i][0]
#       yc = whiteCentertCoordinate[i][1]
#       xo = outerPoint[j][0]
#       yo = outerPoint[j][1]
#       r = (xc - xo) ** 2 + (yc - yo) ** 2
#       r = int(np.sqrt(r))
#       accumulation[r] += 1

# print(accumulation)
# print(sum(accumulation))


# countLabel = [0] * n_2
# for i in range(n_2):
#   for y in range(480):
#     for x in range(640):
#       # (labelNumber_2 = label[1])
#       if labelNumber_2[y][x] == i+1:
#         countLabel[i] += 1

# for i in range(n_2):
#   print("ラベル{0}には画素が{1}個あります".format(i+1,countLabel[i]))

# 中心領域の画素: n個 -> countLabelにそれぞれ格納されている
# バブル外形の輪郭の画素: m個 -> outerPoint[i][2]にラベルが格納されているのでその個数


# for i in range(n_2):
#   labelNumber_2Img = blackImage.copy()
#   labelNumber_2Img[ labelNumber_2 == i, ] = [0, 0, 255]
#   cv2.imshow("{}".format(i+1), labelNumber_2Img)


# for i in range(n_2):
#   print("ラベル{0}は{1}点".format(i+1, countLabel[i]))


# 円の描画
# print(cv2.minEnclosingCircle(contours[2]))
# detectCircle = countourImg.copy()
# for i in range(len(contours)):
#   center,radius = cv2.minEnclosingCircle(contours[i])
#   cv2.circle(detectCircle, (int(center[0]), int(center[1])), int(radius), (0,0,255), thickness=1, lineType=cv2.LINE_8, shift=0)


# cv2.imshow("detectionWhiteRegion3", detectionWhiteRegion3)
# cv2.imshow("輪郭画像", countourImg)
# cv2.imshow('outerLabeled', outerLabeled)  # バブル外形にラベリングしたもの

# cv2.imshow('labelNumber_2Img', labelNumber_2Img)  # バブル外形にラベリングしたもの


# cv2.imshow("円", detectCircle)
# for i in range(len(contours)):
#   if len(contours[i])


# print(hierarchy)
# cv2.imshow("detectionBubble", img)


# cv2.waitKey(0)

# cv2.destroyAllWindows()
