import cv2
import numpy as np
import matplotlib.pyplot as plt


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
cv2.imshow('gray', gray)
cv2.imshow('Laplacian', gray_abs_lap)
cv2.imshow('Laplacian2', lap)
cv2.waitKey(0)





dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
grad = np.sqrt(dx ** 2 + dy ** 2)

#8ビット符号なし整数変換
gray_abs_sobelx = cv2.convertScaleAbs(dx) 
gray_abs_sobely = cv2.convertScaleAbs(dy)
gray_abs_sobel = cv2.convertScaleAbs(grad)

# 結果の出力
cv2.imshow('gray', gray)
cv2.imshow('Sobel_x', gray_abs_sobelx)
cv2.imshow('Sobel_y', gray_abs_sobely)
cv2.imshow('Sobel', gray_abs_sobel)
cv2.waitKey(0)

#  Cannyフィルター
edges = cv2.Canny(gray,100,200)

# 結果の出力
cv2.imshow('gray', gray)
cv2.imshow('edges', edges)
cv2.waitKey(0)


cv2.destroyAllWindows()


