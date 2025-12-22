"""
此子项目用于实现归一化（Normalization）方法中的Z-Score归一化（Z-Score Normalization）
直接读取噪声处理后的CSV文件，无需重新执行噪声识别流程
"""
from scrapy import Spider
import pandas as pd
import numpy as np
import os

class Discussionspider(Spider):
    name = 'discussion_normalize_z_scroe_spiders'
    
    # 核心配置项（读取噪声处理后的文件）
    input_noise_replaced_path = os.path.join('data', 'csv', 'fedora_centos_topics_noise_replaced.csv')  # 噪声处理后的数据路径
    numeric_fields = ['总评论数', '阅读量']  # 需要归一化的数值型字段
    invalid_values = ['未获取到信息', 'N/A', '', ' ', 'NaN', 'None']  # 无效值标识

    def start_requests(self):
        """Scrapy入口：直接加载噪声处理后的数据，执行Z-Score归一化"""
        self.logger.warning("===== 开始执行Z-Score归一化流程（直接使用噪声处理后的数据）=====")
        
        # 1. 加载噪声处理后的数据
        df_noise_replaced = self.load_noise_processed_data()
        if df_noise_replaced is None:
            self.logger.error("噪声处理后的数据加载失败，终止归一化流程")
            return
        
        # 2. 执行Z-Score归一化
        df_normalized = self.z_score_normalization(df_noise_replaced)
        
        # 3. 打印归一化结果（示例数据 + 统计信息）
        self.print_normalization_result(df_normalized)
        
        self.logger.warning("\n===== Z-Score归一化流程完成 =====")
        return

    def load_noise_processed_data(self):
        """加载噪声处理后的CSV文件，预处理无效值"""
        # 检查文件存在性
        if not os.path.exists(self.input_noise_replaced_path):
            self.logger.error(f"噪声处理后的数据文件不存在：{self.input_noise_replaced_path}")
            self.logger.error("请先执行噪声识别与替换流程，生成该文件后再运行归一化")
            return None
        
        # 读取CSV
        try:
            df = pd.read_csv(
                self.input_noise_replaced_path,
                encoding='utf-8-sig'
            )
            self.logger.warning(f"成功读取噪声处理后的数据，共 {len(df)} 条记录，字段：{list(df.columns)}")
            
            # 检查数值型字段是否存在
            missing_numeric_fields = [f for f in self.numeric_fields if f not in df.columns]
            if missing_numeric_fields:
                self.logger.error(f"数据缺失需归一化的数值型字段：{missing_numeric_fields}")
                return None
            
            # 替换无效值为统一标识（便于后续过滤）
            df = df.replace(self.invalid_values, np.nan)
            return df
        
        except Exception as e:
            self.logger.error(f"加载噪声处理后的数据失败：{str(e)}")
            return None

    def z_score_normalization(self, df):
        """Z-Score归一化核心逻辑：对数值型字段进行标准化"""
        df_normalized = df.copy()
        
        for field in self.numeric_fields:
            self.logger.warning(f"\n----- 对字段「{field}」执行Z-Score归一化 -----")
            
            # 提取有效数值（过滤NaN和非数值）
            numeric_series = pd.to_numeric(df_normalized[field], errors='coerce').dropna()
            if len(numeric_series) == 0:
                self.logger.warning(f"  字段「{field}」无有效数值，跳过归一化")
                continue
            
            # 计算均值（μ）和标准差（σ）
            mean_val = numeric_series.mean()
            std_val = numeric_series.std()
            self.logger.warning(f"  原始数据统计：均值={mean_val:.2f}，标准差={std_val:.2f}")
            
            # 处理标准差为0的边界情况（所有数值相同）
            if std_val < 1e-6:
                self.logger.warning(f"  警告：字段「{field}」所有有效数值相同，归一化后均为0")
                def normalize_func(x):
                    if pd.isna(x):
                        return "未获取到信息"  # 无效值还原为原始标识
                    return 0.0
            else:
                # Z-Score公式：z = (x - μ) / σ
                def normalize_func(x):
                    if pd.isna(x):
                        return "未获取到信息"
                    try:
                        val = float(x)
                        return (val - mean_val) / std_val
                    except:
                        return "未获取到信息"
            
            # 执行归一化替换
            df_normalized[field] = df_normalized[field].apply(normalize_func)
            self.logger.warning(f"  归一化完成：有效数值条数={len(numeric_series)}")
        
        return df_normalized

    def print_normalization_result(self, df_normalized):
        """打印归一化结果（示例数据 + 关键统计信息）"""
        self.logger.warning("\n===== Z-Score归一化结果展示 =====")
        
        # 1. 格式化输出：避免科学计数法，保留4位小数
        pd.options.display.float_format = '{:.4f}'.format
        
        # 2. 打印前15条数据示例（展示数值型字段归一化结果 + 发布时间字段）
        display_fields = self.numeric_fields + ['发布时间']  # 保留发布时间用于上下文参考
        self.logger.warning(f"\n【前15条数据示例】")
        for idx, row in df_normalized.head(15).iterrows():
            row_info = f"第{idx+1:2d}条："
            for field in display_fields:
                val = row[field]
                # 对浮点数格式化，其他类型直接显示
                val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                row_info += f"{field}={val_str:12s} | "
            self.logger.warning(row_info.rstrip(" | "))
        
        # 3. 打印归一化后的数据统计（验证标准化效果）
        self.logger.warning(f"\n【归一化后数值统计（理论：均值≈0，标准差≈1）】")
        for field in self.numeric_fields:
            # 提取有效归一化数值
            normalized_series = pd.to_numeric(df_normalized[field], errors='coerce').dropna()
            if len(normalized_series) == 0:
                self.logger.warning(f"  {field}：无有效归一化数据")
                continue
            
            # 计算统计指标
            mean_norm = normalized_series.mean()
            std_norm = normalized_series.std()
            min_norm = normalized_series.min()
            max_norm = normalized_series.max()
            median_norm = normalized_series.median()
            
            self.logger.warning(
                f"  {field}："
                f"均值={mean_norm:.4f} | "
                f"标准差={std_norm:.4f} | "
                f"中位数={median_norm:.4f} | "
                f"范围=[{min_norm:.4f}, {max_norm:.4f}] | "
                f"有效数据量={len(normalized_series)}"
            )
        
        # 4. 打印整体数据概况
        total_records = len(df_normalized)
        self.logger.warning(f"\n【整体概况】")
        self.logger.warning(f"  总记录数：{total_records}")
        self.logger.warning(f"  归一化字段：{self.numeric_fields}")
        self.logger.warning(f"  无效值标识：未获取到信息")