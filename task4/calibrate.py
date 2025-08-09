import cv2
import numpy as np
import os


file_in = 'p1'   # 原始图片存放位置
file_out = 'p3'   # 最后图片的保存位置

w = 8
h = 6

objp = np.zeros((w * h, 3), np.float32)   
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)  
objpoints = [] 
imgpoints = [] 

images = os.listdir(file_in)   # 读入图像序列
i = 0
img_h = 0
img_w = 0

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
for fname in images:
    img = cv2.imread(file_in + '/' + fname)
    img_h = np.size(img, 0)
    img_w = np.size(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if ret == True:
        i += 1
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)  
        imgpoints.append(corners) 
        # 角点显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.imwrite(file_out + '/print_corners' + str(i) + '.jpg', img)
        cv2.waitKey(10)
cv2.destroyAllWindows()

"""
求解参数
输入：世界坐标系里的位置；像素坐标；图像的像素尺寸大小；
输出：
ret: 重投影误差；
mtx：内参矩阵；
dist：畸变系数；
rvecs：旋转向量 （外参数）；
tvecs：平移向量 （外参数）；
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(("ret（重投影误差）:"), ret)
print(("mtx（内参矩阵）:\n"), mtx)
print(("dist（畸变参数）:\n"), dist)  # 5个畸变参数，(k_1,k_2,p_1,p_2,k_3)
print(("rvecs（旋转向量）:\n"), rvecs)
print(("tvecs（平移向量）:\n"), tvecs)


# 优化内参数和畸变系数
# 使用相机内参mtx和畸变系数dist，并使用cv.getOptimalNewCameraMatrix()
# 通过设定自由自由比例因子alpha。
# 当alpha设为0的时候，将会返回一个剪裁过的将去畸变后不想要的像素去掉的内参数和畸变系数；
# 当alpha设为1的时候，将会返回一系个包含额外黑色像素点的内参数和畸变数，并返回一个ROI用于将其剪裁掉。
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (img_w, img_h), 0, (img_w, img_h))


# 矫正畸变
img2 = cv2.imread(file_in + '/5.jpg')
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
cv2.imwrite(file_out + '/calibresult.jpg', dst)
print("newcameramtx（优化后相机内参）:\n", newcameramtx)

# 反投影误差total_error,越接近0，说明结果越理想。
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)  
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)  
    total_error += error
print(("total error: "), total_error / len(objpoints))   
