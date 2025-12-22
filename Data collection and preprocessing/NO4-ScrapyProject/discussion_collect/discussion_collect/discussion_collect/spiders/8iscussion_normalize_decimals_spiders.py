"""
此子项目用于实现归一化（Normalization）方法中的小数定标归一化（Decimal Scaling Normalization）
直接使用噪声处理后的数据，仅打印输出结果，不新建文件
"""
from scrapy import Spider
import pandas as pd
import numpy as np
import os

class DiscussionSpider(Spider):
    name = 'discussion_normalize_decimals_spiders'
    
    # 核心配置项
    input_noise_replaced_path = os.path.join('data', 'csv', 'fedora_centos_topics_noise_replaced.csv')  # 噪声处理后的数据路径
    numeric_fields = ['总评论数', '阅读量']  # 需要归一化的数值型字段
    invalid_values = ['未获取到信息', 'N/A', '', ' ', 'NaN', 'None']  # 无效值标识

    def start_requests(self):
        """Scrapy入口：加载噪声后数据→执行小数定标归一化→打印结果"""
        self.logger.info("===== 开始执行小数定标归一化流程（使用噪声处理后的数据）=====")
        
        # 1. 加载噪声处理后的数据
        df_noise_replaced = self.load_noise_processed_data()
        if df_noise_replaced is None:
            self.logger.error("数据加载失败，终止归一化流程")
            return
        
        # 2. 执行小数定标归一化
        df_normalized, scaling_info = self.decimal_scaling_normalization(df_noise_replaced)
        
        # 3. 打印归一化结果（示例 + 统计信息）
        self.print_normalization_result(df_normalized, scaling_info)
        
        self.logger.info("\n===== 小数定标归一化流程完成 =====")
        return

    def load_noise_processed_data(self):
        """加载噪声处理后的CSV文件，预处理无效值"""
        # 检查文件存在性
        if not os.path.exists(self.input_noise_replaced_path):
            self.logger.error(f"噪声处理后的数据文件不存在：{self.input_noise_replaced_path}")
            self.logger.error("请先执行噪声识别与替换流程，生成该文件后再运行")
            return None
        
        # 读取CSV
        try:
            df = pd.read_csv(
                self.input_noise_replaced_path,
                encoding='utf-8-sig'
            )
            self.logger.info(f"成功读取噪声处理后的数据，共 {len(df)} 条记录，字段：{list(df.columns)}")
            
            # 检查目标数值字段是否存在
            missing_fields = [f for f in self.numeric_fields if f not in df.columns]
            if missing_fields:
                self.logger.error(f"数据缺失需归一化的字段：{missing_fields}")
                return None
            
            # 替换无效值为NaN（便于过滤）
            df = df.replace(self.invalid_values, np.nan)
            return df
        
        except Exception as e:
            self.logger.error(f"加载数据失败：{str(e)}")
            return None

    def decimal_scaling_normalization(self, df):
        """小数定标归一化核心逻辑：x' = x / 10^j（使数据落在[-1, 1]区间）"""
        df_normalized = df.copy()
        scaling_info = {}  # 存储每个字段的缩放信息（最大绝对值、j值）
        
        for field in self.numeric_fields:
            self.logger.info(f"\n----- 对字段「{field}」执行小数定标归一化 -----")
            
            # 步骤1：提取有效数值（过滤NaN和非数值）
            numeric_series = pd.to_numeric(df_normalized[field], errors='coerce').dropna()
            if len(numeric_series) == 0:
                self.logger.info(f"  字段「{field}」无有效数值，跳过归一化")
                scaling_info[field] = {'max_abs': None, 'j': None}
                continue
            
            # 步骤2：计算最大绝对值和缩放系数j（10的幂次）
            max_abs = numeric_series.abs().max()  # 最大绝对值
            if max_abs == 0:
                # 所有有效数值均为0，归一化后仍为0
                j = 1
                self.logger.info(f"  所有有效数值均为0，归一化后保持为0")
            else:
                # j为满足 10^(j-1) ≤ max_abs < 10^j 的最小整数
                j = int(np.ceil(np.log10(max_abs))) if max_abs >= 1 else 0
            scaling_factor = 10 ** j  # 缩放因子：10^j
            
            # 存储缩放信息
            scaling_info[field] = {
                'max_abs': max_abs,
                'j': j,
                'scaling_factor': scaling_factor
            }
            
            # 步骤3：打印原始数据和缩放参数
            self.logger.info(f"  原始数据统计：最大绝对值={max_abs:.2f}")
            self.logger.info(f"  缩放参数：j={j}，缩放因子=10^{j}={scaling_factor}")
            
            # 步骤4：执行归一化
            def normalize_func(x):
                if pd.isna(x):
                    return "未获取到信息"  # 无效值还原为原始标识
                try:
                    val = float(x)
                    return val / scaling_factor  # 小数定标公式
                except:
                    return "未获取到信息"
            
            df_normalized[field] = df_normalized[field].apply(normalize_func)
            self.logger.info(f"  归一化完成：有效数值条数={len(numeric_series)}")
        
        return df_normalized, scaling_info

    def print_normalization_result(self, df_normalized, scaling_info):
        """打印归一化结果（示例数据 + 关键统计信息）"""
        self.logger.info("\n===== 小数定标归一化结果展示 =====")
        
        # 1. 格式化输出：保留4位小数，避免科学计数法
        pd.options.display.float_format = '{:.4f}'.format
        
        # 2. 打印前15条数据示例（包含数值型字段 + 发布时间，便于上下文参考）
        display_fields = self.numeric_fields + ['发布时间']
        self.logger.info(f"\n【前15条数据示例】")
        for idx, row in df_normalized.head(15).iterrows():
            row_str = f"第{idx+1:2d}条："
            for field in display_fields:
                val = row[field]
                # 格式化数值显示
                val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                row_str += f"{field}={val_str:12s} | "
            self.logger.info(row_str.rstrip(" | "))
        
        # 3. 打印每个字段的归一化统计信息
        self.logger.info(f"\n【归一化后详细统计】")
        for field in self.numeric_fields:
            # 提取有效归一化数值
            normalized_series = pd.to_numeric(df_normalized[field], errors='coerce').dropna()
            if len(normalized_series) == 0:
                self.logger.info(f"  {field}：无有效归一化数据")
                continue
            
            # 计算统计指标
            min_norm = normalized_series.min()
            max_norm = normalized_series.max()
            mean_norm = normalized_series.mean()
            std_norm = normalized_series.std()
            
            # 提取缩放信息
            info = scaling_info[field]
            self.logger.info(
                f"  {field}："
                f"缩放因子=10^{info['j']}={info['scaling_factor']} | "
                f"范围=[{min_norm:.4f}, {max_norm:.4f}]（理论∈[-1,1]） | "
                f"均值={mean_norm:.4f} | "
                f"标准差={std_norm:.4f} | "
                f"有效数据量={len(normalized_series)}"
            )
        
        # 4. 整体概况
        total_records = len(df_normalized)
        valid_counts = {}
        for field in self.numeric_fields:
            valid_counts[field] = len(pd.to_numeric(df_normalized[field], errors='coerce').dropna())
        
        self.logger.info(f"\n【整体概况】")
        self.logger.info(f"  总记录数：{total_records}")
        self.logger.info(f"  归一化字段：{self.numeric_fields}")
        self.logger.info(f"  各字段有效数据量：{valid_counts}")
        self.logger.info(f"  无效值标识：未获取到信息")