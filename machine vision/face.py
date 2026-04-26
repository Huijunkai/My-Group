import cv2
import numpy as np
import sqlite3
import os
import glob
import time

# Optional EXIF reader: prefer installed package but degrade gracefully
try:
    import exifread
    HAS_EXIFREAD = True
except Exception:
    exifread = None
    HAS_EXIFREAD = False

# ====================== 数据库相关函数（保持不变） ======================
def init_face_database():
    """初始化人脸数据库，创建face_info表，打印数据库绝对路径"""
    code_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(code_dir, 'face_database.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            face_feature TEXT NOT NULL,
            image_path TEXT
        )
    ''')
    # 如果旧库没有 image_path 列，尝试添加（向后兼容）
    try:
        cursor.execute("PRAGMA table_info(face_info)")
        cols = [r[1] for r in cursor.fetchall()]
        if 'image_path' not in cols:
            cursor.execute('ALTER TABLE face_info ADD COLUMN image_path TEXT')
    except Exception:
        pass
    
    conn.commit()
    conn.close()
    print(f"人脸数据库初始化成功！数据库路径：{db_path}")

def save_face_to_database(username, face_feature, image_path=None):
    """将人脸特征存入数据库"""
    code_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(code_dir, 'face_database.db')
    
    try:
        if isinstance(face_feature, str):
            feature_str = face_feature
        elif face_feature is None:
            feature_str = ''
        else:
            feature_str = ','.join(map(str, face_feature))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO face_info (username, face_feature, image_path) VALUES (?, ?, ?)',
                      (username, feature_str, image_path if image_path is not None else ''))
        conn.commit()
        conn.close()
        print(f"用户{username}人脸信息入库成功！")
        return True
    except sqlite3.IntegrityError:
        print(f"错误：用户名{username}已存在！")
        return False
    except Exception as e:
        print(f"入库失败：{e}")
        return False

def load_face_from_database():
    """从数据库加载所有人脸信息"""
    code_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(code_dir, 'face_database.db')
    
    face_list = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT username, face_feature, image_path FROM face_info')
        results = cursor.fetchall()

        for username, feature_str, image_path in results:
            try:
                face_feature = np.array(list(map(float, feature_str.split(',')))) if feature_str else None
                face_list.append((username, face_feature, image_path))
            except Exception:
                face_list.append((username, None, image_path))
        
        conn.close()
    except Exception as e:
        print(f"加载人脸信息失败：{e}")
    
    return face_list

# ====================== 人脸检测相关函数（增强拍照适配） ======================
def skin_color_segmentation(frame):
    """改进的肤色分割：组合 YCrCb + HSV，适配拍照光照变化"""
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    # YCrCb 分量阈值
    ycrcb = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    cr_mask = cv2.inRange(cr, 110, 190)
    cb_mask = cv2.inRange(cb, 60, 140)
    ycrcb_mask = cv2.bitwise_and(cr_mask, cb_mask)
    # HSV 分量阈值
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
    hsv_mask1 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
    hsv_mask2 = cv2.inRange(hsv, np.array([165, 20, 70]), np.array([180, 255, 255]))
    hsv_mask = cv2.bitwise_or(hsv_mask1, hsv_mask2)
    # 形态学处理
    skin_mask = cv2.bitwise_or(ycrcb_mask, hsv_mask)
    kernel = np.ones((3, 3), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.medianBlur(skin_mask, 5)
    return skin_mask

def detect_face_region(frame):
    """精准人脸定位（适配拍照场景，优先Haar级联）"""
    # 优先使用Haar级联检测（拍照场景更稳定）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))  # 增大最小人脸尺寸
    
    if len(faces) > 0:
        # 选取最大人脸（拍照时通常只有单人脸）
        areas = [w*h for (x, y, w, h) in faces]
        idx = int(np.argmax(areas))
        x, y, w_face, h_face = faces[idx]
        # 扩展区域，保留完整面部特征
        x = max(0, x - 15)
        y = max(0, y - 25)
        w_face = min(frame.shape[1] - x, w_face + 30)
        h_face = min(frame.shape[0] - y, h_face + 30)
        face_roi = frame[y:y+h_face, x:x+w_face]
        return (x, y, w_face, h_face), face_roi
    
    # Haar检测失败时，使用肤色分割后备方案
    skin_mask = skin_color_segmentation(frame)
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    h, w = frame.shape[:2]
    min_face_area = h * w * 0.05  # 拍照场景增大最小人脸面积，过滤噪声
    max_face_area = h * w * 0.7
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_face_area < area < max_face_area:
            x, y, w_face, h_face = cv2.boundingRect(contour)
            aspect_ratio = w_face / h_face
            if 0.5 < aspect_ratio < 1.8:
                valid_contours.append((contour, area))
    
    if not valid_contours:
        return None, None
    
    max_contour, _ = max(valid_contours, key=lambda x: x[1])
    x, y, w_face, h_face = cv2.boundingRect(max_contour)
    x = max(0, x - 15)
    y = max(0, y - 25)
    w_face = min(frame.shape[1] - x, w_face + 30)
    h_face = min(frame.shape[0] - y, h_face + 30)
    face_roi = frame[y:y+h_face, x:x+w_face]
    return (x, y, w_face, h_face), face_roi

def normalize_face_roi(face_roi, target_size=(128, 128)):
    """人脸归一化（增强拍照图片适配）"""
    normalized_roi = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_CUBIC)
    gray_roi = cv2.cvtColor(normalized_roi, cv2.COLOR_BGR2GRAY)
    # 双重直方图均衡化，提升拍照图片对比度
    gray_roi = cv2.equalizeHist(gray_roi)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_roi = clahe.apply(gray_roi)
    gray_roi = cv2.normalize(gray_roi, None, 0, 255, cv2.NORM_MINMAX)
    return gray_roi

# ====================== 人脸特征提取相关函数（保持不变） ======================
def cal_gradient(gray_img):
    """计算图像梯度"""
    gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) * (180 / np.pi)
    angle[angle < 0] += 180
    return magnitude, angle

def lbp_feature_extraction(gray_img, radius=1, neighbors=8):
    """LBP特征提取（简化计算，适配普通拍照图片）"""
    h, w = gray_img.shape
    lbp_img = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)
    
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = gray_img[i, j]
            lbp_code = 0
            for k in range(neighbors):
                theta = 2 * np.pi * k / neighbors
                x = int(i + 