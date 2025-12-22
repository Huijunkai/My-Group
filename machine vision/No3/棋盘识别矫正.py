import cv2
import numpy as np
import os

# 定义棋盘格的行数和列数
ROWS = 23
COLS = 17

# 定义世界坐标系中的棋盘格角点
objp = np.zeros((ROWS * COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)

# 存储所有图像中的对象点和图像点
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# 加载图像路径
images = [
    'machine vision/data/distort_images/1.bmp',
    'machine vision/data/distort_images/2.bmp',
    'machine vision/data/distort_images/3.bmp',
    'machine vision/data/distort_images/4.bmp',
    'machine vision/data/distort_images/5.bmp',
    'machine vision/data/distort_images/6.bmp',
    'machine vision/data/distort_images/7.bmp',
    'machine vision/data/distort_images/8.bmp',
    'machine vision/data/distort_images/9.bmp',
    'machine vision/data/distort_images/10.bmp',
    'machine vision/data/distort_images/11.bmp',
    'machine vision/data/distort_images/12.bmp',
    'machine vision/data/distort_images/13.bmp',
    'machine vision/data/distort_images/14.bmp',
    'machine vision/data/distort_images/15.bmp',
    'machine vision/data/distort_images/16.bmp',
    'machine vision/data/distort_images/17.bmp'
]

# 创建结果保存目录
save_dir = 'camera_calibration_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"警告：无法读取图像 {fname}，跳过该图像。")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 绘制角点并保存
        cv2.drawChessboardCorners(img, (COLS, ROWS), corners, ret)
        cv2.imwrite(os.path.join(save_dir, f"corners_{os.path.basename(fname)}"), img)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 标定相机
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 保存标定结果
np.savez(os.path.join(save_dir, 'calibration_results.npz'), mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print("相机矩阵:")
print(mtx)
print("\n畸变系数:")
print(dist)

# ============== 新增：图像矫正与批量处理 ==============

if ret:
    # 批量矫正所有图像
    for idx, fname in enumerate(images):
        img_to_undistort = cv2.imread(fname)
        if img_to_undistort is None:
            print(f"错误：无法读取图像 {fname} 用于矫正。")
            continue
        h, w = img_to_undistort.shape[:2]

        # 计算新的最优相机矩阵
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # 执行图像矫正
        dst = cv2.undistort(img_to_undistort, mtx, dist, None, newcameramtx)

        # 裁剪黑边（可选，根据需求决定是否裁剪）
        x, y, w_roi, h_roi = roi
        if all([x, y, w_roi, h_roi]):  # 确保ROI有效
            dst = dst[y:y+h_roi, x:x+w_roi]
            img_to_undistort_cropped = img_to_undistort[y:y+h_roi, x:x+w_roi]
            combined = np.hstack((img_to_undistort_cropped, dst))
            cv2.imwrite(os.path.join(save_dir, f"comparison_{idx+1}.bmp"), combined)
        else:
            cv2.imwrite(os.path.join(save_dir, f"undistorted_{idx+1}.bmp"), dst)

    # 打印分析结果
    print("\n图像矫正分析：")
    print("1. 矫正后的图像直线边缘（如棋盘格）更平直，有效修正了桶形/枕形畸变。")
    print("2. 若存在黑边，是为保证无失真的裁剪结果；可通过调整`getOptimalNewCameraMatrix`的`alpha`参数权衡失真与黑边。")
    print("3. 矫正后图像几何结构更贴合真实世界，可提升后续视觉任务（如测量、识别）的精度。")
else:
    print("标定失败，无法进行图像矫正。")