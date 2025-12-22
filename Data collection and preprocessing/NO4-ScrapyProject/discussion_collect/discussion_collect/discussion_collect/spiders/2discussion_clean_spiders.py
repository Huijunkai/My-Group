"""
此爬虫项目用于对Fedora/CentOS官方讨论区爬取的热门话题数据进行清洗，
包括处理空值、重复值、日期格式转换、数值字段类型转换及特殊字符过滤等操作。
清洗后的数据将保存为新的CSV文件，确保数据质量和一致性，便于后续分析使用。 
"""
import csv
import os
import locale
import re
from datetime import datetime
from tqdm import tqdm  # 新增：导入tqdm进度条库
from scrapy import Spider

class DiscussionSpider(Spider):
    name = 'discussion_clean_spiders'
    
    # 输入输出文件路径配置
    input_csv_path = os.path.join('data', 'csv', 'fedora_centos_topics.csv')
    output_csv_path = os.path.join('data', 'csv', 'fedora_centos_topics_cleaned.csv')
    # 发布时间字段名（根据你的CSV实际字段名修改！）
    publish_time_field = '发布时间'
    
    # ===== 新增：数值字段转换配置 =====
    # 需转换的数值字段（键：字段名，值：目标类型 int/float）
    numeric_fields = {
        '总评论数': int,    # 评论数转整数
        '阅读量': float    # 阅读量转浮点（也可改为int）
    }
    # 数值转换时的无效值（跳过这些值的转换）
    numeric_invalid_values = ['未获取到信息', 'NaN', 'None', '', ' ', 'N/A', '暂无数据']
    
    # ===== 新增：特殊字符过滤配置 =====
    # 允许保留的字符正则表达式（中文、字母、数字、常见标点、空格、小数点）
    ALLOWED_CHARS_PATTERN = r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.,，。！？；;:""''（）\(\)-_=+\[\]{}、·…—]'
    # 统计特殊字符过滤次数
    special_char_remove_count = 0

    def start_requests(self):
        """启动数据清洗流程"""
        self.logger.info("开始数据预处理...")
        self.clean_csv_data()
        self.logger.info("数据预处理完成")
        return  # 不发起网络请求
    
    # ===== 新增：特殊字符过滤函数 =====
    def remove_special_chars(self, text):
        """
        过滤文本中的特殊字符，仅保留允许的字符
        :param text: 原始文本
        :return: 过滤后的文本
        """
        if not text or text == "未获取到信息":
            return text
        
        # 确保输入是字符串
        if not isinstance(text, str):
            text = str(text)
        
        # 记录过滤前的文本长度（用于统计）
        original_length = len(text)
        
        # 过滤特殊字符
        cleaned_text = re.sub(self.ALLOWED_CHARS_PATTERN, '', text)
        
        # 统计过滤次数（只要有字符被移除就算一次）
        if len(cleaned_text) < original_length:
            self.special_char_remove_count += 1
        
        return cleaned_text.strip() or "未获取到信息"
    
    def convert_date_format(self, date_str):
        """
        转换日期格式为「YYYY年M月D日」（兼容英文月份格式）
        :param date_str: 原始日期字符串
        :return: 转换后的日期字符串 / 未获取到信息（失败时）
        """
        # 空值直接返回预设值
        if not date_str or str(date_str).strip() in ['', '未获取到信息']:
            return "未获取到信息"
        
        # 配置本地化（兼容英文月份解析，适配Windows/Linux/Mac）
        try:
            # Windows系统
            locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
        except:
            try:
                # Linux/Mac系统
                locale.setlocale(locale.LC_TIME, 'en_US')
            except:
                # 兜底（部分系统无需手动设置）
                pass
        
        # 扩展日期格式模板（新增英文月份格式）
        date_formats = [
            # 数字格式（原有）
            '%Y-%m-%d', '%Y/%m/%d', '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S', '%Y年%m月%d日', '%m/%d/%Y',
            '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M',
            # 英文月份格式（新增核心）
            '%B %d, %Y',    # February 7, 2022（无前导0）
            '%B %d,%Y',     # February 7,2022（无空格）
            '%B %02d, %Y',  # February 07, 2022（有前导0）
            '%b %d, %Y',    # Feb 7, 2022（缩写月份）
            '%b %02d, %Y'   # Feb 07, 2022（缩写+前导0）
        ]
        
        # 尝试解析日期
        clean_date = str(date_str).strip()
        for fmt in date_formats:
            try:
                dt = datetime.strptime(clean_date, fmt)
                # 格式化为「YYYY年M月D日」（M/D不带前导0）
                return f"{dt.year}年{dt.month}月{dt.day}日"
            except (ValueError, TypeError):
                continue
        
        # 所有格式都匹配失败（保留日志提示）
        self.logger.info(f"日期格式解析失败，原始值：{date_str}")
        return "未获取到信息"
    
    # ===== 新增：数值清洗与转换辅助函数 =====
    def clean_numeric_string(self, value):
        """清洗数值字符串：仅保留数字和小数点，去除其他字符（如逗号、单位、空格）"""
        if not isinstance(value, str):
            value = str(value)
        # 保留0-9和.，去除所有其他字符
        cleaned = ''.join([c for c in value.strip() if c.isdigit() or c == '.'])
        # 处理多个小数点的情况（仅保留第一个）
        if cleaned.count('.') > 1:
            parts = cleaned.split('.')
            cleaned = parts[0] + '.' + ''.join(parts[1:])
        return cleaned if cleaned else None
    
    def safe_convert_numeric(self, value, target_type):
        """安全转换数值类型，失败时返回默认值"""
        # 跳过无效值
        if value in self.numeric_invalid_values:
            return 0 if target_type == int else 0.0
        
        # 清洗数值字符串
        cleaned_val = self.clean_numeric_string(value)
        if not cleaned_val:
            self.logger.info(f"数值清洗失败，原始值：{value} → 设为默认值")
            return 0 if target_type == int else 0.0
        
        # 尝试转换类型
        try:
            converted = target_type(cleaned_val)
            return converted
        except (ValueError, TypeError):
            self.logger.info(f"数值转换失败（{target_type.__name__}），原始值：{value} → 设为默认值")
            return 0 if target_type == int else 0.0

    def clean_csv_data(self):
        """处理CSV文件中的空值、重复值、日期格式 + 新增数值字段类型转换 + 特殊字符过滤"""
        # 重置特殊字符过滤计数器
        self.special_char_remove_count = 0
        
        # 检查输入文件是否存在
        if not os.path.exists(self.input_csv_path):
            self.logger.error(f"输入文件不存在: {self.input_csv_path}")
            return
        
        # 读取原始数据
        try:
            with open(self.input_csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                # 检查发布时间字段是否存在
                if self.publish_time_field not in headers:
                    self.logger.error(f"CSV中未找到「{self.publish_time_field}」字段，请检查字段名！")
                    return
                # 检查数值字段是否存在
                missing_numeric_fields = [f for f in self.numeric_fields if f not in headers]
                if missing_numeric_fields:
                    self.logger.error(f"CSV中缺失数值字段：{missing_numeric_fields}")
                    return
                data = list(reader)
            self.logger.info(f"成功读取原始数据，共 {len(data)} 条记录")
        except Exception as e:
            self.logger.error(f"读取CSV文件失败: {str(e)}")
            return
        
        # ========== 核心修改：初始化tqdm进度条 ==========
        # 样式和前序爬虫保持一致（绿色、宽度80、描述清晰）
        pbar = tqdm(
            total=len(data),
            desc="数据清洗进度",
            unit="条",
            ncols=80,
            colour="green",
            leave=True
        )
        
        # 处理空值 + 日期格式 + 数值转换 + 特殊字符过滤 + 精准统计
        cleaned_data = []
        null_replace_count = 0  # 空值替换计数器
        date_convert_fail_count = 0  # 日期转换失败计数器
        # 新增：数值转换统计
        numeric_convert_stats = {f: {'success': 0, 'fail': 0} for f in self.numeric_fields}
        
        for row in data:
            cleaned_row = {}
            for key, value in row.items():
                # 统一处理空值
                str_value = str(value).strip() if value is not None else ''
                if str_value == '' or str_value.lower() in ['nan', 'none']:
                    cleaned_row[key] = "未获取到信息"
                    null_replace_count += 1
                else:
                    # ===== 新增：过滤特殊字符 =====
                    cleaned_value = self.remove_special_chars(str_value)
                    cleaned_row[key] = cleaned_value
            
            # 单独处理发布时间的日期格式
            original_time = cleaned_row[self.publish_time_field]
            converted_time = self.convert_date_format(original_time)
            if converted_time == "未获取到信息" and original_time != "未获取到信息":
                date_convert_fail_count += 1
            cleaned_row[self.publish_time_field] = converted_time
            
            # ===== 数值字段类型转换 =====
            for numeric_field, target_type in self.numeric_fields.items():
                original_numeric = cleaned_row[numeric_field]
                # 安全转换数值
                converted_numeric = self.safe_convert_numeric(original_numeric, target_type)
                # 更新统计
                if converted_numeric in [0, 0.0] and original_numeric not in self.numeric_invalid_values:
                    numeric_convert_stats[numeric_field]['fail'] += 1
                else:
                    numeric_convert_stats[numeric_field]['success'] += 1
                # 替换为转换后的值
                cleaned_row[numeric_field] = converted_numeric
            
            cleaned_data.append(cleaned_row)
            
            # ========== 核心修改：更新进度条 ==========
            pbar.update(1)
        
        # ========== 核心修改：关闭进度条 ==========
        pbar.close()
        
        # 处理重复值
        seen = set()
        unique_data = []
        for row in cleaned_data:
            # 转换数值为字符串后再去重（避免类型问题）
            row_tuple = tuple(str(row[header]) for header in headers)
            if row_tuple not in seen:
                seen.add(row_tuple)
                unique_data.append(row)
        
        # 统计输出
        duplicate_count = len(cleaned_data) - len(unique_data)
        self.logger.info(f"空值处理完成，共替换 {null_replace_count} 处空值（已标注为未获取到信息）")
        self.logger.info(f"日期格式转换完成，共 {date_convert_fail_count} 条记录转换失败(已从英语转换为中文格式)")
        # 新增：特殊字符过滤统计
        self.logger.info(f"特殊字符过滤完成，共处理 {self.special_char_remove_count} 条包含特殊字符的记录（已过滤@#$%^&*等特殊字符）")
        # 数值转换统计
        for field, stats in numeric_convert_stats.items():
            total = stats['success'] + stats['fail']
            self.logger.info(f"【{field}】数值转换完成：成功 {stats['success']} 条，失败 {stats['fail']} 条（总计 {total} 条）")
        self.logger.info(f"重复值处理完成，共移除 {duplicate_count} 条重复记录")
        self.logger.info(f"清洗后剩余 {len(unique_data)} 条有效记录")
        
        # 保存清洗后的数据
        try:
            with open(self.output_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(unique_data)
            self.logger.info(f"清洗后的数据已保存至: {os.path.abspath(self.output_csv_path)}")
        except Exception as e:
            self.logger.error(f"保存清洗后的数据失败: {str(e)}")