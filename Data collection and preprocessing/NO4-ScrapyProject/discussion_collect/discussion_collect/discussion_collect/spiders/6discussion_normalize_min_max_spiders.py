"""
此子项目：仅执行最小-最大归一化（读取已噪声处理的文件，不重复噪声处理）
核心特性：
1. 直接读取噪声处理后的CSV文件，跳过重复噪声处理
2. 对数值型字段执行Min-Max归一化（缩放到[0,1]）
3. 不生成任何文件，仅在控制台打印归一化结果和统计信息
"""
from scrapy import Spider
import pandas as pd
import numpy as np
import os

class Discussionspider(Spider):
    name = 'discussion_normalize_min_max_spiders'
    
    # 核心配置项（直接读取噪声处理后文件）
    input_noise_processed_path = os.path.join('data', 'csv', 'fedora_centos_topics_noise_replaced.csv')  # 噪声处理后的输入文件
    numeric_fields = ['总评论数', '阅读量']  # 需要归一化的数值型字段
    # 归一化参数
    min_max_range = (0, 1)  # 归一化目标区间
    invalid_value_placeholder = '未获取到信息'  # 与噪声处理文件一致的无效值标识
    print_max_rows = 20  # 最终数据打印最大行数（避免输出过长）

    def start_requests(self):
        """Scrapy入口：启动归一化流程（仅读取噪声处理后文件）"""
        self.logger.warning("===== 开始执行：仅归一化（读取噪声处理后文件，不重复噪声处理） =====")
        
        # 1. 加载噪声处理后的数据
        df_noise_processed = self.load_noise_processed_data()
        if df_noise_processed is None:
            self.logger.error("数据加载失败，终止归一化流程")
            return
        
        # 2. 执行最小-最大归一化（仅数值型字段）
        df_final, normalize_stats = self.process_min_max_normalization(df_noise_processed)
        
        # 3. 打印归一化统计报告
        self.print_normalize_stats(normalize_stats)
        
        # 4. 打印最终归一化后的数据结果
        self.print_final_data(df_final)
        
        self.logger.warning("\n===== 归一化流程完成（无文件生成，仅打印结果） =====")
        return

    def load_noise_processed_data(self):
        """加载已完成噪声处理的文件，验证数据有效性"""
        # 检查噪声处理文件存在性
        if not os.path.exists(self.input_noise_processed_path):
            self.logger.error(f"噪声处理后的文件不存在：{self.input_noise_processed_path}")
            self.logger.error("请先执行噪声处理流程，生成该文件后再运行")
            return None
        
        # 读取CSV文件
        try:
            df = pd.read_csv(
                self.input_noise_processed_path,
                encoding='utf-8-sig'
            )
            self.logger.warning(f"成功读取噪声处理后的数据，共 {len(df)} 条记录，字段：{list(df.columns)}")
            
            # 验证数值型字段是否存在
            missing_fields = [f for f in self.numeric_fields if f not in df.columns]
            if missing_fields:
                self.logger.error(f"文件缺失需归一化的数值型字段：{missing_fields}")
                return None
            
            # 统计各数值型字段的无效值数量
            self.logger.warning("\n【字段无效值统计】")
            for field in self.numeric_fields:
                invalid_count = (df[field] == self.invalid_value_placeholder).sum()
                valid_count = len(df) - invalid_count
                self.logger.warning(f"  • {field}：有效数值数={valid_count} 条，无效值数={invalid_count} 条")
            
            return df
        
        except Exception as e:
            self.logger.error(f"加载噪声处理后文件失败：{str(e)}")
            return None

    def process_min_max_normalization(self, df):
        """执行最小-最大归一化：x' = (x - min) / (max - min) * (target_max - target_min) + target_min"""
        df_copy = df.copy()
        target_min, target_max = self.min_max_range
        
        # 初始化归一化统计信息
        normalize_stats = {
            field: {
                'original_min': None,
                'original_max': None,
                'normalized_min': target_min,
                'normalized_max': target_max,
                'valid_count': 0,
                'invalid_count': 0,
                'mean_after_norm': None  # 归一化后的均值
            } for field in self.numeric_fields
        }
        
        self.logger.warning("\n===== 开始执行最小-最大归一化 =====")
        for field in self.numeric_fields:
            self.logger.warning(f"\n----- 处理字段：{field} -----")
            
            # 步骤1：分离有效数值和无效值（过滤无效值标识）
            valid_mask = (df_copy[field] != self.invalid_value_placeholder)
            # 转换为数值类型，过滤无法转换的异常值
            valid_series = pd.to_numeric(df_copy.loc[valid_mask, field], errors='coerce').dropna()
            
            # 统计有效/无效数量
            normalize_stats[field]['valid_count'] = len(valid_series)
            normalize_stats[field]['invalid_count'] = len(df_copy) - len(valid_series)
            
            # 无有效数值时，跳过该字段归一化
            if len(valid_series) == 0:
                self.logger.warning(f"  无有效数值，跳过归一化")
                continue
            
            # 步骤2：计算原始数据的极值（噪声处理后的极值）
            original_min = valid_series.min()
            original_max = valid_series.max()
            normalize_stats[field]['original_min'] = original_min
            normalize_stats[field]['original_max'] = original_max
            self.logger.warning(f"  噪声处理后原始范围：[{original_min:.2f}, {original_max:.2f}]")
            
            # 步骤3：处理极值相同的情况（避免除零错误）
            if original_max == original_min:
                self.logger.warning(f"  所有有效值均为 {original_min}，归一化为目标区间最小值 {target_min}")
                df_copy.loc[valid_mask, field] = target_min
                normalize_stats[field]['mean_after_norm'] = target_min
                continue
            
            # 步骤4：执行归一化计算
            def normalize_value(x):
                if x == self.invalid_value_placeholder:
                    return self.invalid_value_placeholder  # 保留无效值标识
                try:
                    val = float(x)
                    # 归一化公式
                    normalized_val = (val - original_min) / (original_max - original_min) * (target_max - target_min) + target_min
                    return round(normalized_val, 6)  # 保留6位小数，避免精度冗余
                except:
                    return self.invalid_value_placeholder  # 无法转换的视为无效值
            
            # 应用归一化
            df_copy[field] = df_copy[field].apply(normalize_value)
            
            # 计算归一化后的均值
            norm_valid_series = pd.to_numeric(df_copy.loc[valid_mask, field], errors='coerce').dropna()
            normalize_stats[field]['mean_after_norm'] = norm_valid_series.mean() if len(norm_valid_series) > 0 else 0
            
            self.logger.warning(f"  归一化完成，目标区间：[{target_min}, {target_max}]")
            self.logger.warning(f"  归一化后均值：{normalize_stats[field]['mean_after_norm']:.6f}")
        
        return df_copy, normalize_stats

    def print_normalize_stats(self, normalize_stats):
        """打印详细的归一化统计报告"""
        self.logger.warning("\n" + "="*60)
        self.logger.warning("===== 归一化统计报告 =====")
        self.logger.warning("="*60)
        
        for field, stats in normalize_stats.items():
            self.logger.warning(f"\n【{field}】")
            if stats['original_min'] is None:
                self.logger.warning(f"  状态：未执行归一化（无有效数值）")
            else:
                self.logger.warning(f"  噪声处理后原始范围：[{stats['original_min']:.2f}, {stats['original_max']:.2f}]")
                self.logger.warning(f"  归一化目标范围：[{stats['normalized_min']}, {stats['normalized_max']}]")
                self.logger.warning(f"  归一化后均值：{stats['mean_after_norm']:.6f}")
            self.logger.warning(f"  有效数值数：{stats['valid_count']} 条")
            self.logger.warning(f"  无效数值数：{stats['invalid_count']} 条")

    def print_final_data(self, df_final):
        """打印归一化后的最终数据（概览+前N条详情）"""
        self.logger.warning("\n" + "="*60)
        self.logger.warning(f"===== 归一化后最终数据结果（共 {len(df_final)} 条） =====")
        self.logger.warning("="*60)
        
        # 1. 数据基本信息
        self.logger.warning(f"\n【数据基本信息】")
        self.logger.warning(f"总记录数：{len(df_final)}")
        self.logger.warning(f"字段列表：{list(df_final.columns)}")
        
        # 2. 数值型字段归一化后统计量（均值、最值、标准差）
        self.logger.warning(f"\n【数值型字段详细统计（归一化后）】")
        for field in self.numeric_fields:
            valid_series = pd.to_numeric(df_final[field], errors='coerce').dropna()
            if len(valid_series) > 0:
                self.logger.warning(f"  • {field}：")
                self.logger.warning(f"    - 均值：{valid_series.mean():.6f}")
                self.logger.warning(f"    - 最小值：{valid_series.min():.6f}")
                self.logger.warning(f"    - 最大值：{valid_series.max():.6f}")
                self.logger.warning(f"    - 标准差：{valid_series.std():.6f}")
                self.logger.warning(f"    - 中位数：{valid_series.median():.6f}")
            else:
                self.logger.warning(f"  • {field}：无有效归一化数值")
        
        # 3. 打印前N条数据详情（美化格式）
        print_rows = min(self.print_max_rows, len(df_final))
        self.logger.warning(f"\n【前 {print_rows} 条数据详情】")
        # 转换为字符串格式统一显示，控制字段宽度
        df_print = df_final.head(print_rows).astype(str)
        print(df_print.to_string(index=False, max_colwidth=25))  # max_colwidth调整字段显示宽度
        self.logger.warning(f"\n注：仅显示前 {print_rows} 条数据，完整数据共 {len(df_final)} 条")