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

# Cannyフィルター
edges = cv2.Canny(gray,100,200)

cv2.imshow('gray', gray)
cv2.imshow('Laplacian', gray_abs_lap)
cv2.imshow('Laplacian2', lap)
cv2.imshow('edges', edges)
cv2.waitKey(0)

cv2.destroyAllWindows()