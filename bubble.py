import cv2
import numpy as np
import matplotlib.pyplot as plt

# 標準出力の色を変える
class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RETURN = '\033[07m' #反転
    ACCENT = '\033[01m' #強調
    FLASH = '\033[05m' #点滅
    RED_FLASH = '\033[05;41m' #赤背景+点滅
    END = '\033[0m'

# グラフ練習

# x = np.random.normal(50,10,1000)

# print(x)

# plt.plot(x)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
print(pycolor.BLUE+'fig.axes:'+pycolor.END, fig.axes)
print(pycolor.BLUE+'ax.figure:'+pycolor.END, ax.figure)
print(pycolor.BLUE+'ax.xaxis:'+pycolor.END, ax.xaxis)
print(pycolor.BLUE+'ax.yaxis:'+pycolor.END, ax.yaxis)
print(pycolor.BLUE+'ax.xaxis.axes:'+pycolor.END, ax.xaxis.axes)
print(pycolor.BLUE+'ax.yaxis.axes:'+pycolor.END, ax.yaxis.axes)
print(pycolor.BLUE+'ax.xaxis.figure:'+pycolor.END, ax.xaxis.figure)
print(pycolor.BLUE+'ax.yaxis.figure:'+pycolor.END, ax.yaxis.figure)
print(pycolor.BLUE+'fig.xaxis:'+pycolor.END, fig.xaxis)

# 入力画像を読み込み
img = cv2.imread("87.jpg")
#cv2.imshow('color', img)
#cv2.waitKey(0)

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray,(5,5),0)
lap = cv2.Laplacian(gray, cv2.CV_32F)
#8ビット符号なし整数変換
gray_abs_lap = cv2.convertScaleAbs(lap) 

# 結果の出力
# cv2.imshow('gray', gray)
# cv2.imshow('Laplacian', gray_abs_lap)
# cv2.imshow('Laplacian2', lap)
# cv2.waitKey(0)


# 動的に変数を作るのは推奨されていない
names = ["aaa", "bbbb", "ccccc"]

# for name in names:
#   x = len(name)
#   exec('{} = {}'.format(name, x))

# print(aaa) # 3
# print(type(aaa)) # int 
# print(str(aaa)) # 3
# print(type(str(aaa))) # str
# print(bbbb) # 4
# print(ccccc) # 5



# print(center+[7])

# select_gravity.append([center, 8])
# select_gravity.append(center2)
# select_gravity.append(center3)



# print(select_gravity)

dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
grad = np.sqrt(dx ** 2 + dy ** 2)

#8ビット符号なし整数変換
gray_abs_sobelx = cv2.convertScaleAbs(dx) 
gray_abs_sobely = cv2.convertScaleAbs(dy)
gray_abs_sobel = cv2.convertScaleAbs(grad)

# 結果の出力
# cv2.imshow('gray', gray)
# cv2.imshow('Sobel_x', gray_abs_sobelx)
# cv2.imshow('Sobel_y', gray_abs_sobely)
# cv2.imshow('Sobel', gray_abs_sobel)
# cv2.waitKey(0)

#  Cannyフィルター
edges = cv2.Canny(gray,150,200)

# 結果の出力
# cv2.imshow('gray', gray)
# cv2.imshow('edges', edges)
# cv2.waitKey(0)


# cv2.destroyAllWindows()


