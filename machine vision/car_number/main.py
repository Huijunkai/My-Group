import os
import sys
import unicodedata
import os.path

current_dir = os.path.dirname(os.path.abspath(__file__))
plate_dir = current_dir
sys.path.append(plate_dir)

from plate_recognition import *

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_folder = sys.argv[1]
    else:
        target_folder = os.path.join(plate_dir, "data_jpg")  # 图片文件价录
    results = batch_process(target_folder)
    # 表格宽度设置
    COLS = {
        'name': 12,
        'plate': 14,
        'color': 10,
        'type': 14,
        'location': 14,
        'status': 24
    }
    """返回字符串的显示宽度，中文等宽字符计为2，其他字符计为1。"""
    def disp_len(s):
        if s is None:
            return 0
        s = str(s)
        l = 0
        for ch in s:
            ea = unicodedata.east_asian_width(ch)
            l += 2 if ea in ('F', 'W') else 1
        return l
    """按显示宽度截断字符串，超出时以省略号结尾（省略号算1宽）。"""
    def truncate_display(s, w):
        if s is None:
            return ''
        s = str(s)
        cur = 0
        out = ''
        for ch in s:
            ea = unicodedata.east_asian_width(ch)
            ch_w = 2 if ea in ('F', 'W') else 1
            if cur + ch_w > w:
                return out + '…'
            out += ch
            cur += ch_w
        return out

    def fmt_fixed(s, w):
        if s is None:
            s = ''
        s = str(s)
        t = truncate_display(s, w)
        pad = w - disp_len(t)
        if pad > 0:
            return t + ' ' * pad
        return t

    sep = ' | '
    header = sep.join([
        fmt_fixed('图片名称', COLS['name']),
        fmt_fixed('识别结果', COLS['plate']),
        fmt_fixed('颜色', COLS['color']),
        fmt_fixed('类型', COLS['type']),
        fmt_fixed('所属地区', COLS['location']),
        fmt_fixed('状态', COLS['status'])
    ])

    total_width = sum(COLS.values()) + len(sep) * 5
    print('=' * total_width)
    print(header)
    print('-' * total_width)
    for res in results:
        print(sep.join([
            fmt_fixed(res.get('图片名称') or '', COLS['name']),
            fmt_fixed(res.get('识别结果') or '', COLS['plate']),
            fmt_fixed(res.get('车牌颜色') or '', COLS['color']),
            fmt_fixed(res.get('车牌类型') or '', COLS['type']),
            fmt_fixed(res.get('所属地区') or '', COLS['location']),
            fmt_fixed(res.get('状态') or '', COLS['status'])
        ]))

    # 状态判断
    def is_success(st):
        return isinstance(st, str) and st.startswith('识别成功')

    success_count = sum(1 for r in results if is_success(r.get('状态')))
    color_stats = {}
    type_stats = {}
    for res in results:
        if is_success(res.get('状态')):
            color_stats[res.get('车牌颜色', '未知')] = color_stats.get(res.get('车牌颜色', '未知'), 0) + 1
            type_stats[res.get('车牌类型', '未知')] = type_stats.get(res.get('车牌类型', '未知'), 0) + 1

    print('\n统计结果:')
    print(f"总图片数：{len(results)}")
    print(f"识别成功率：{(success_count/len(results) if len(results)>0 else 0):.1%}")
    print('\n颜色分布:')
    if success_count == 0:
        print('  无识别成功结果')
    else:
        max_color_len = max((len(c) for c in color_stats.keys()), default=4)
        for color, count in color_stats.items():
            pct = (count / success_count) if success_count > 0 else 0
            print(f"  {color.ljust(max_color_len)} : {str(count).rjust(3)} 张  ({pct:.1%})")
    print('\n车辆类型:')
    if success_count == 0:
        print('  无识别成功结果')
    else:
        max_type_len = max((len(t) for t in type_stats.keys()), default=4)
        for ptype, count in type_stats.items():
            pct = (count / success_count) if success_count > 0 else 0
            print(f"  {ptype.ljust(max_type_len)} : {str(count).rjust(3)} 辆  ({pct:.1%})")