import os
import re
import numpy as np
import glob
import cv2
from hyperlpr import HyperLPR_plate_recognition
from plate_mappings import PROVINCE_MAP, SPECIAL_PLATE_TYPES
# 解决 OpenCV 兼容性问题
if not hasattr(cv2, 'estimateRigidTransform'):
    def _estimateRigidTransform(src, dst, fullAffine=False):
        M, inliers = cv2.estimateAffinePartial2D(src, dst)
        return M
    cv2.estimateRigidTransform = _estimateRigidTransform
# 解决numpy兼容问题
if not hasattr(np, 'int'):
    np.int = np.int32
# 调试开关
DEBUG = True
# 常见车牌字符串后处理
def postprocess_plate(plate_str):
    if not plate_str:
        return plate_str
    s = re.sub(r'[\s·-]', '', plate_str).upper()
    allowed_chinese = set(PROVINCE_MAP.keys()) | set(['警', '学', '领', '使', '港', '澳', '台'])
    out = []
    for i, ch in enumerate(s):
        # 首位若为中文且在省份表中，保留
        if i == 0 and ch in allowed_chinese:
            out.append(ch)
            continue
        # 保留 ASCII 字母和数字
        if '0' <= ch <= '9' or 'A' <= ch <= 'Z':
            out.append(ch)
            continue
        # 保留一些中文特殊字
        if ch in allowed_chinese:
            out.append(ch)
            continue
        # 其它字符丢弃
    return ''.join(out)

# 常见 OCR 误识别位置感知修正 字符替换o I 为 0 1
def correct_plate_string(plate_str):
    if not plate_str:
        return plate_str
    if not plate_str:
        return plate_str
    s = plate_str.upper()
    chars = list(s)
    # 常见混淆字符到数字的映射
    mapping = {
        'O': '0', 'I': '1'
    }
    n = len(chars)
    # 从第二位开始对尾部做保守替换，避免改动省份简称
    start = max(1, n - 5)
    for i in range(start, n):
        c = chars[i]
        if c in mapping:
            chars[i] = mapping[c]

    # 若数字个数仍然不足，放宽替换范围（从第二位开始）
    digit_count = sum(ch.isdigit() for ch in chars)
    if digit_count < 2:
        for i in range(1, n):
            c = chars[i]
            if c in mapping:
                chars[i] = mapping[c]

    return ''.join(chars)
# 基于颜色空间分析车牌颜色
def get_plate_color(image, plate_region):
    if not plate_region or all(v == 0 for v in plate_region):
        return "未知"

    x1, y1, x2, y2 = map(int, plate_region)
    plate_img = image[y1:y2, x1:x2]
    if plate_img.size == 0:
        return "未知"

    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    pixel_count = plate_img.shape[0] * plate_img.shape[1]

    # 颜色阈值范围
    PLATE_COLOR_RANGES = {
        'blue': [(90, 40, 40), (130, 255, 255)],
        'yellow': [(15, 60, 60), (40, 255, 255)],
        'green': [(35, 30, 30), (95, 255, 255)],
        'red1': [(0, 60, 50), (10, 255, 255)],
        'red2': [(160, 60, 50), (180, 255, 255)],
    }

    def mask_ratio(hsv_img, low, up):
        low = np.array(low, dtype=np.uint8)
        up = np.array(up, dtype=np.uint8)
        mask = cv2.inRange(hsv_img, low, up)
        return cv2.countNonZero(mask) / float(pixel_count)

    ratios = {}
    ratios['blue'] = mask_ratio(hsv, *PLATE_COLOR_RANGES['blue'])
    ratios['yellow'] = mask_ratio(hsv, *PLATE_COLOR_RANGES['yellow'])
    ratios['green'] = mask_ratio(hsv, *PLATE_COLOR_RANGES['green'])
    red_ratio = mask_ratio(hsv, *PLATE_COLOR_RANGES['red1']) + mask_ratio(hsv, *PLATE_COLOR_RANGES['red2'])

    # 白色判断：S 低且 V 高
    _, s_channel, v_channel = cv2.split(hsv)
    mean_s = float(np.mean(s_channel))
    mean_v = float(np.mean(v_channel))
    is_white = (mean_s <= 60 and mean_v >= 180)
    # 新增黑色判断：S 低且 V 低
    is_black = (mean_s <= 20 and mean_v <= 40)

    # 判定优先级：新能源绿 -> 蓝 -> 黄 -> 白 -> 红 -> 兜底
    if ratios.get('green', 0.0) >= 0.18:
        return '新能源浅绿'
    if is_black:
        return '黑色'  # 新增黑色返回
    if ratios.get('blue', 0.0) >= 0.31:
        return '蓝色'
    if ratios.get('yellow', 0.0) >= 0.18:
        return '黄色'
    if is_white:
        return '白色'
    if red_ratio >= 0.12:
        return '红色'


    # 兜底：返回占比最大的颜色（若超过低阈值）
    candidates = {'blue': ratios.get('blue', 0.0), 'yellow': ratios.get('yellow', 0.0),
                  'green': ratios.get('green', 0.0), 'red': red_ratio}
    main = max(candidates, key=candidates.get)
    if candidates.get(main, 0.0) >= 0.12:
        mapping = {'blue': '蓝色', 'yellow': '黄色', 'green': '绿色', 'red': '红色'}
        return mapping.get(main, '未知')

    return '未知'

# 判断是否为左黄右绿的新能源大型车牌
def is_split_yellow_green_plate(image, plate_region):
    if not plate_region:
        return False
    x1, y1, x2, y2 = map(int, plate_region)
    plate_img = image[y1:y2, x1:x2]
    if plate_img.size == 0:
        return False

    h, w = plate_img.shape[:2]
    # 左侧宽度比例（根据常见大型新能源车牌样式设定）
    left_ratio = 0.18
    left_w = max(1, int(w * left_ratio))

    left_img = plate_img[:, :left_w]
    right_img = plate_img[:, left_w:]

    hsv_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2HSV)
    hsv_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2HSV)

    # 黄色与绿色阈值
    low_y, up_y = np.array([15, 40, 40], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8)
    low_g, up_g = np.array([35, 30, 30], dtype=np.uint8), np.array([95, 255, 255], dtype=np.uint8)

    mask_l_y = cv2.inRange(hsv_l, low_y, up_y)
    mask_r_g = cv2.inRange(hsv_r, low_g, up_g)

    ratio_l_y = cv2.countNonZero(mask_l_y) / float(left_img.shape[0] * left_img.shape[1])
    ratio_r_g = cv2.countNonZero(mask_r_g) / float(right_img.shape[0] * right_img.shape[1])

    # 阈值可调整：左侧黄色占比至少 0.4，右侧绿色占比至少 0.25
    if ratio_l_y >= 0.40 and ratio_r_g >= 0.25:
        return True
    return False

def enhance_image_for_plate(image):
    """图像增强：对可能的车牌颜色区域做小幅度增强，返回增强后的 BGR 图像。
    增强策略保守，避免破坏原始字符信息。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 简化并参数化增强：对蓝/绿/红区域做轻微的 S/V 提升
    color_masks = {
        'blue': (np.array([90, 40, 40], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8)),
        'green': (np.array([35, 40, 40], dtype=np.uint8), np.array([100, 255, 255], dtype=np.uint8)),
        'red': (np.array([0, 100, 100], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
    }

    # 提升参数
    s_increase = {'blue': 30, 'green': 0, 'red': 0}
    v_increase = {'blue': 0, 'green': 20, 'red': 30}

    for cname, (low, up) in color_masks.items():
        mask = cv2.inRange(hsv, low, up)
        if s_increase.get(cname, 0) > 0:
            s[mask > 0] = np.clip(s[mask > 0] + s_increase[cname], 0, 255)
        if v_increase.get(cname, 0) > 0:
            v[mask > 0] = np.clip(v[mask > 0] + v_increase[cname], 0, 255)

    enhanced_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

# 预处理函数: CLAHE + 锐化
def preprocess_for_recognition(image):
    # 转 HSV，增强 V
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # CLAHE 在 V 通道
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v_clahe = clahe.apply(v)
    except Exception:
        v_clahe = v

    hsv_clahe = cv2.merge([h, s, v_clahe])
    img_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

    # 锐化：unsharp mask
    gaussian = cv2.GaussianBlur(img_clahe, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img_clahe, 1.5, gaussian, -0.5, 0)

    return sharpened

# 格式检查 新能源车辆（2+6） 蓝牌（2+5）
def check_plate_number_format(plate_str, plate_color):
    if not plate_str:
        return False, None, "空车牌字符串"
    s = plate_str.upper()
    length = len(s)
    if plate_color == "新能源浅绿" or "新能源" in plate_str:
        return (length == 8), "2+6(总长8)", ("" if length == 8 else f"长度 {length} != 8")
    if plate_color == "蓝色":
        return (length == 7), "2+5(总长7)", ("" if length == 7 else f"长度 {length} != 7")
    return True, None, ""
# 核心检测函数，支持多尺度检测
def _detect_plates_hyperlpr(img_name,image, scales=(1.0, 0.8, 1.2)):
    all_results = []
    for scale in scales:
        try:
            if scale != 1.0:
                resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            else:
                resized = image
            try:
                res = HyperLPR_plate_recognition(resized) or []
            except Exception as e:
                if DEBUG:
                    print(f"HyperLPR(array) 调用异常(scale={scale}): {e}")
                res = []
        except Exception:
            res = []

        if DEBUG:
            try:
                print(f"图片名称={img_name},检测车牌（使用hyperlpr技术）: 缩放比例={scale}, 识别结果={len(res)}")
            except Exception:
                pass

        for r in res:
            plate = r[0] if len(r) > 0 else ''
            conf = r[1] if len(r) > 1 else 0.0
            bbox = r[2] if len(r) > 2 else None
            all_results.append({'plate': plate, 'conf': conf, 'bbox': bbox, 'scale': scale})
    return all_results

# 映射检测框坐标到原始图像尺寸
def _map_bbox_to_original(bbox, scale, img_shape):
    if not bbox:
        return None
    try:
        bx = list(map(float, bbox))
        x1, y1, x2, y2 = map(int, bx)
        if scale and scale != 1.0:
            x1 = int(round(x1 / scale))
            y1 = int(round(y1 / scale))
            x2 = int(round(x2 / scale))
            y2 = int(round(y2 / scale))
        h, w = img_shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
    except Exception:
        return None

# 主识别函数
def recognize_plate(image_path):
    if not os.path.exists(image_path):
        return None, "文件不存在", None, None

    image = cv2.imread(image_path)
    if image is None:
        return None, "无法读取图片", None, None

    enhanced_img = enhance_image_for_plate(image)
    img_name = os.path.basename(image_path)

    try:
        # 多尺度检测
        candidates = _detect_plates_hyperlpr(img_name,enhanced_img, scales=(1.0, 0.8, 1.2))
        if not candidates:
            # 回退：仅尝试以图像数组调用 hyperlpr（路径调用在某些版本会导致类型错误）
            direct = []
            try:
                if HyperLPR_plate_recognition is None:
                    if DEBUG:
                        print("HyperLPR_plate_recognition 未导入，跳过回退调用")
                else:
                    if DEBUG:
                        print(f"主检测未命中，尝试直接用 HyperLPR_plate_recognition(image)")
                    direct = HyperLPR_plate_recognition(enhanced_img) or []
            except Exception as e:
                direct = []
                if DEBUG:
                    print(f"direct(array) 调用异常: {e}")

            if direct:
                if DEBUG:
                    print(f"回退检测命中 {len(direct)} 条结果")
                candidates = []
                for r in direct:
                    plate = r[0] if len(r) > 0 else ''
                    conf = r[1] if len(r) > 1 else 0.0
                    bbox = r[2] if len(r) > 2 else None
                    candidates.append({'plate': plate, 'conf': conf, 'bbox': bbox, 'scale': 1.0})

            if not candidates:
                try:
                    dbg_dir = os.path.join(os.path.dirname(image_path), 'debug_outputs')
                    os.makedirs(dbg_dir, exist_ok=True)
                    base = os.path.splitext(os.path.basename(image_path))[0]
                    orig_p = os.path.join(dbg_dir, base + '_orig.jpg')
                    enh_p = os.path.join(dbg_dir, base + '_enhanced.jpg')
                    cv2.imwrite(orig_p, image)
                    cv2.imwrite(enh_p, enhanced_img)
                    if DEBUG:
                        print(f"未检测到车牌，已保存调试图像到: {dbg_dir}")
                except Exception as e:
                    if DEBUG:
                        print(f"保存调试图像失败: {e}")

                return None, "未识别到车牌", None, None

        # 取置信度最高
        best = max(candidates, key=lambda x: float(x.get('conf', 0.0)))

        plate_str = postprocess_plate(best.get('plate', '') or '')
        plate_str = correct_plate_string(plate_str)

        mapped_bbox = _map_bbox_to_original(best.get('bbox'), best.get('scale', 1.0), enhanced_img.shape)

        # 先判断是否为左黄右绿的新能源大型车牌
        if mapped_bbox and is_split_yellow_green_plate(enhanced_img, mapped_bbox):
            plate_type = "新能源大型客车"
            plate_color = '新能源绿黄'
        else:
            plate_color = get_plate_color(enhanced_img, mapped_bbox)

            # 特殊车牌类型判断
            plate_type = "普通车牌"
            # 军用白牌：以两个拉丁大写字母开头（例如 KZ12345）
            if re.match(r'^[A-Z]{2}\d+', plate_str):
                plate_type = "军用白牌"
            else:
                for char, type_name in SPECIAL_PLATE_TYPES.items():
                    if char in plate_str:
                        plate_type = type_name
                        break

        if plate_type == "军用白牌" or (isinstance(plate_type, str) and "警" in plate_type):
            plate_color = "白色"

        valid_format, expected_fmt, fmt_note = check_plate_number_format(plate_str, plate_color)
        if not valid_format:
            return None, "识别失败(异常车牌)", None, None

        return plate_str, "识别成功", plate_color, plate_type
    except Exception as e:
        return None, f"识别错误：{str(e)}", None, None

def batch_process(folder_path):
    """批量处理文件夹中的图片"""
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_paths:
        print("未找到图片文件！")
        return []

    results = []
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        plate, status, color, plate_type = recognize_plate(img_path)
        # 所属地区判定
        if plate_type == "军用白牌":
            location = "军区"
        else:
            if plate:
                first = plate[0]
                if first in PROVINCE_MAP:
                    location = PROVINCE_MAP.get(first, "未知")
                elif re.match(r'^[A-Z]', plate):
                    location = "外牌/特殊"
                else:
                    location = "未知"
            else:
                location = "未知"

        # 根据颜色与特殊标记映射为用户要求的类型标签
        display_type = plate_type or "未知"
        c = (color or "").lower()
        p = plate or ""
        # 蓝牌 -> 小型客车
        if '蓝' in c:
            display_type = '小型客车'
        # 黄牌 -> 大型客车，若包含 '学' 则为教练车
        elif '黄' in c:
            if '学' in p:
                display_type = '教练车'
            else:
                display_type = '大型客车'
        # 白牌 -> 军用车辆或警用车辆
        elif '白' in c or plate_type == '军用白牌' or '军用' in (plate_type or ''):
            if '警' in p or ('警' in (plate_type or '')):
                display_type = '警用车辆'
            else:
                display_type = '军用车辆'
        # 黑色 -> 港澳黑牌车辆（含港/澳字符）
        elif '黑' in c:
            if '港' in p or '澳' in p:
                display_type = '港澳黑牌车辆'
            else:
                display_type = '特殊黑牌车辆'
        # 新能源分为小型/大型：若 earlier 识别为新能源大型车辆 则为新能源大型客车
        elif plate_type == '新能源大型车辆' or '新能源大型' in (plate_type or ''):
            display_type = '新能源大型客车'
        elif '新能源' in (color or '') or '绿' in c:
            display_type = '新能源小型客车'

        results.append({
            "图片名称": img_name,
            "识别结果": plate or "未识别",
            "车牌颜色": color or "未知",
            "车牌类型": display_type,
            "所属地区": location,
            "状态": status
        })
    return results