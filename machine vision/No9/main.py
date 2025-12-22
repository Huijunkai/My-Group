import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

try:
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
except:
    font = FontProperties(family="SimHei")

def load_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) 
    return img, gray

def original_harris(gray_img, k=0.04, threshold=0.01):
    """
    原始Harris算法实现
    参数：
    - k: 响应函数参数（建议0.04-0.06）
    - threshold: 角点响应阈值（值越大，检测越严格）
    """
    # 计算图像梯度
    dx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度乘积并高斯平滑
    dx2 = cv2.GaussianBlur(np.square(dx), (5, 5), 1.5)
    dy2 = cv2.GaussianBlur(np.square(dy), (5, 5), 1.5)
    dxy = cv2.GaussianBlur(np.multiply(dx, dy), (5, 5), 1.5)
    
    # 计算角点响应函数 R = det(M) - k*(trace(M))^2
    det_m = dx2 * dy2 - dxy * dxy
    trace_m = dx2 + dy2
    r = det_m - k * np.square(trace_m)
    
    # 阈值筛选角点
    corner_mask = r > threshold * r.max()
    return corner_mask

def multi_scale_harris(gray_img, k=0.04, threshold=0.01, scales=[1.0, 1.5, 2.0]):
    """
    多尺度Harris算法实现（通过多尺度融合筛选稳定角点）
    参数：
    - scales: 不同尺度的高斯标准差列表
    """
    height, width = gray_img.shape
    all_corners = np.zeros((height, width), dtype=bool)
    
    for sigma in scales:
        # 不同尺度下的高斯平滑
        blurred = cv2.GaussianBlur(gray_img, (0, 0), sigmaX=sigma)
        
        # 计算梯度与梯度乘积
        dx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        dx2 = cv2.GaussianBlur(np.square(dx), (5, 5), 1.5)
        dy2 = cv2.GaussianBlur(np.square(dy), (5, 5), 1.5)
        dxy = cv2.GaussianBlur(np.multiply(dx, dy), (5, 5), 1.5)
        
        # 计算响应函数并筛选角点
        det_m = dx2 * dy2 - dxy * dxy
        trace_m = dx2 + dy2
        r = det_m - k * np.square(trace_m)
        scale_corners = r > threshold * r.max()
        
        # 累加各尺度角点（取交集，保留稳定角点）
        all_corners = np.logical_and(all_corners, scale_corners) if sigma != scales[0] else scale_corners
    
    return all_corners

def visualize_results(img, original_corners, multi_scale_corners, image_name):
    img_original = img.copy()
    img_multi_scale = img.copy()
    
    # 绘制角点（红色圆点，半径2）
    y, x = np.where(original_corners)
    for xi, yi in zip(x, y):
        cv2.circle(img_original, (xi, yi), 2, (0, 0, 255), -1)
    
    y, x = np.where(multi_scale_corners)
    for xi, yi in zip(x, y):
        cv2.circle(img_multi_scale, (xi, yi), 2, (0, 0, 255), -1)
    
    # 转换颜色空间（OpenCV→Matplotlib）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_multi_scale = cv2.cvtColor(img_multi_scale, cv2.COLOR_BGR2RGB)
    
    # 绘图布局
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('原始图像', fontproperties=font)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_original)
    plt.title('原始Harris角点检测', fontproperties=font)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_multi_scale)
    plt.title('多尺度Harris角点检测', fontproperties=font)
    plt.axis('off')
    
    plt.suptitle(f'{image_name} 角点检测结果对比', fontproperties=font, fontsize=14)
    plt.savefig('machine vision/No9/harris_comparison_cn.png', dpi=300, bbox_inches='tight')  # 保存结果图像
    plt.show()


if __name__ == "__main__":
    image_path = "machine vision/No9/fig_house.tif"  
    img, gray_img = load_image(image_path)
    
    # 执行两种算法
    original_corners = original_harris(gray_img, k=0.04, threshold=0.01)
    multi_scale_corners = multi_scale_harris(gray_img, k=0.04, threshold=0.01, scales=[1.0, 1.5, 2.0])
    
    # 可视化结果
    visualize_results(img, original_corners, multi_scale_corners, "房屋图像")
    
    # 输出角点数量统计
    original_count = np.sum(original_corners)
    multi_scale_count = np.sum(multi_scale_corners)
    print(f"原始Harris算法检测到的角点数量:{original_count}")
    print(f"多尺度Harris算法检测到的角点数量:{multi_scale_count}")