"""  
此子项目的主要内容是对清洗后的文件中的数值进行噪声识别与替换，
采用IQR原则对数值型字段进行噪声识别，采用频率
原则对类别型字段进行噪声识别，
均采用众数进行替换。
"""
from scrapy import Spider
import pandas as pd
import numpy as np
import os
from collections import Counter

class DiscussionNoiseSpider(Spider):
    name = 'discussion_noise_replace_spiders'
    
    # 核心配置项（适配实际CSV字段）
    input_csv_path = os.path.join('data', 'csv', 'fedora_centos_topics_cleaned.csv')
    output_csv_path = os.path.join('data', 'csv', 'fedora_centos_topics_noise_replaced.csv')
    # 仅保留实际存在的字段
    numeric_fields = ['总评论数', '阅读量']  # 数值型字段（CSV中存在）
    categorical_fields = ['发布时间']  # 类别型字段（仅保留发布时间，删除标题类型）
    # 噪声判断阈值
    iqr_outlier_threshold = 1.5  # 数值型IQR阈值
    categorical_freq_threshold = 0.01  # 类别型频率阈值（<1%视为噪声）
    invalid_values = ['未获取到信息', 'N/A', '', ' ', 'NaN', 'None']

    def start_requests(self):
        """Scrapy入口：启动噪声识别与众数替换流程"""
        self.logger.warning("===== 开始执行噪声数据识别与众数替换 =====")
        
        # 1. 加载并预处理数据
        df = self.load_and_preprocess_data()
        if df is None:
            self.logger.error("数据加载失败，终止噪声处理流程")
            return
        
        # 2. 初始化噪声替换统计
        noise_stats = {
            'numeric': {field: {'noise_count': 0, 'mode': None} for field in self.numeric_fields},
            'categorical': {field: {'noise_count': 0, 'mode': None} for field in self.categorical_fields}
        }
        
        # 3. 处理数值型字段：IQR判噪声 + 众数替换
        df_numeric_clean = self.process_numeric_noise(df, noise_stats)
        
        # 4. 处理类别型字段：频率判噪声 + 众数替换
        df_final = self.process_categorical_noise(df_numeric_clean, noise_stats)
        
        # 5. 打印噪声处理统计
        self.print_noise_stats(noise_stats)
        
        # 6. 保存处理后的数据
        self.save_processed_data(df_final)
        
        self.logger.warning("\n===== 噪声数据处理完成 =====")
        return  # 无网络请求，直接返回

    def load_and_preprocess_data(self):
        """加载Clean数据，预处理无效值"""
        # 检查文件存在性
        if not os.path.exists(self.input_csv_path):
            self.logger.error(f"Clean文件不存在：{self.input_csv_path}")
            return None
        
        # 读取CSV
        try:
            df = pd.read_csv(
                self.input_csv_path,
                encoding='utf-8-sig',
                dtype=str  # 先统一读为字符串，避免类型错误
            )
            self.logger.warning(f"成功读取Clean数据，共 {len(df)} 条记录，字段：{list(df.columns)}")
            
            # 检查目标字段是否存在
            all_target_fields = self.numeric_fields + self.categorical_fields
            missing_fields = [f for f in all_target_fields if f not in df.columns]
            if missing_fields:
                self.logger.error(f"CSV缺失目标字段：{missing_fields}")
                return None
            
            # 替换无效值为统一标识（不参与噪声判断）
            df = df.replace(self.invalid_values, np.nan)
            return df
        
        except Exception as e:
            self.logger.error(f"加载数据失败：{str(e)}")
            return None

    def process_numeric_noise(self, df, noise_stats):
        """处理数值型字段：IQR原则识别噪声，众数替换"""
        df_copy = df.copy()
        for field in self.numeric_fields:
            self.logger.warning(f"\n----- 处理数值型字段：{field} -----")
            
            # 步骤1：提取有效数值数据（过滤NaN）
            numeric_series = pd.to_numeric(df_copy[field], errors='coerce').dropna()
            if len(numeric_series) == 0:
                self.logger.warning(f"字段「{field}」无有效数值，跳过噪声处理")
                noise_stats['numeric'][field]['mode'] = np.nan
                continue
            
            # 步骤2：计算IQR和异常值边界
            q1 = numeric_series.quantile(0.25)
            q3 = numeric_series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_outlier_threshold * iqr
            upper_bound = q3 + self.iqr_outlier_threshold * iqr
            self.logger.warning(f"  IQR边界：[{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # 步骤3：计算众数（处理多众数，取第一个）
            mode_val = numeric_series.mode().iloc[0] if not numeric_series.mode().empty else 0
            noise_stats['numeric'][field]['mode'] = mode_val
            self.logger.warning(f"  众数：{mode_val}")
            
            # 步骤4：识别并替换噪声值
            def replace_numeric_noise(x):
                if pd.isna(x):
                    return np.nan  # 无效值保留NaN，后续不统计
                try:
                    val = float(x)
                    if val < lower_bound or val > upper_bound:
                        noise_stats['numeric'][field]['noise_count'] += 1
                        return mode_val  # 众数替换噪声
                    return val
                except:
                    return np.nan  # 非数值保留NaN
            
            df_copy[field] = df_copy[field].apply(replace_numeric_noise)
            self.logger.warning(f"  识别出噪声数据：{noise_stats['numeric'][field]['noise_count']} 条")
        
        return df_copy

    def process_categorical_noise(self, df, noise_stats):
        """处理类别型字段：频率原则识别噪声，众数替换"""
        df_copy = df.copy()
        for field in self.categorical_fields:
            self.logger.warning(f"\n----- 处理类别型字段：{field} -----")
            
            # 步骤1：提取有效类别数据（过滤NaN）
            cat_series = df_copy[field].dropna()
            if len(cat_series) == 0:
                self.logger.warning(f"字段「{field}」无有效类别，跳过噪声处理")
                noise_stats['categorical'][field]['mode'] = np.nan
                continue
            
            # 步骤2：计算类别频率
            total_valid = len(cat_series)
            freq_counter = Counter(cat_series)
            freq_dict = {k: v/total_valid for k, v in freq_counter.items()}
            self.logger.warning(f"  高频类别前5：{dict(list(freq_dict.items())[:5])}")
            
            # 步骤3：计算众数（出现频率最高的类别）
            mode_val = max(freq_counter, key=freq_counter.get)
            noise_stats['categorical'][field]['mode'] = mode_val
            self.logger.warning(f"  众数：{mode_val}（频率：{freq_dict[mode_val]:.2%}）")
            
            # 步骤4：识别并替换噪声值（频率<threshold）
            def replace_categorical_noise(x):
                if pd.isna(x):
                    return np.nan  # 无效值保留NaN
                if freq_dict.get(x, 0) < self.categorical_freq_threshold:
                    noise_stats['categorical'][field]['noise_count'] += 1
                    return mode_val  # 众数替换噪声
                return x
            
            df_copy[field] = df_copy[field].apply(replace_categorical_noise)
            self.logger.warning(f"  识别出噪声数据：{noise_stats['categorical'][field]['noise_count']} 条")
        
        # 将NaN还原为"未获取到信息"
        df_copy = df_copy.replace(np.nan, "未获取到信息")
        return df_copy

    def print_noise_stats(self, noise_stats):
        """打印噪声处理统计结果"""
        self.logger.warning("\n===== 噪声数据处理统计 ======")
        # 数值型字段统计
        self.logger.warning("【数值型字段（IQR原则）】")
        for field, stats in noise_stats['numeric'].items():
            mode = stats['mode']
            mode_str = f"{mode:.2f}" if isinstance(mode, (int, float)) else "无"
            self.logger.warning(f"  {field}：噪声数={stats['noise_count']}，替换众数={mode_str}")
        
        # 类别型字段统计
        self.logger.warning("【类别型字段（频率原则）】")
        for field, stats in noise_stats['categorical'].items():
            mode_str = stats['mode'] if pd.notna(stats['mode']) else "无"
            self.logger.warning(f"  {field}：噪声数={stats['noise_count']}，替换众数={mode_str}")

    def save_processed_data(self, df):
        """保存噪声替换后的数据"""
        try:
            df.to_csv(
                self.output_csv_path,
                index=False,
                encoding='utf-8-sig'
            )
            self.logger.warning(f"\n噪声处理后的数据已保存至：{os.path.abspath(self.output_csv_path)}")
        except Exception as e:
            self.logger.error(f"保存数据失败：{str(e)}")