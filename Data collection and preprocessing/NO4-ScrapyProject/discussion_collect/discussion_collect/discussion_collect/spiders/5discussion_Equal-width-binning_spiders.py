"""
此子项目的主要内容是对清洗后的文件中的数值进行等宽分箱处理，
并对分箱结果进行均值/中值/边界值平滑处理，
以便后续的数据分析与建模使用。
"""
import os
import numpy as np
import pandas as pd
from scrapy import Spider

class DiscussionManageSpider(Spider):
    name = 'discussion_Equal-width-binning_spiders'
    
    # 文件路径与分箱配置
    input_csv_path = os.path.join('data', 'csv', 'fedora_centos_topics_cleaned.csv')
    bins = 4  # 等宽分箱数量
    # 目标处理字段（根据实际CSV字段名调整）
    target_fields = ['总评论数', '阅读量']
    
    def start_requests(self):
        """启动数据分箱处理流程"""
        self.logger.warning("开始处理数值型字段的等宽分箱...")
        self.process_binning()
        self.logger.warning("数值分箱处理完成")
        return  # 不发起网络请求
    
    def process_binning(self):
        """读取清洗后的数据并对目标字段进行等宽分箱处理"""
        # 检查文件是否存在
        if not os.path.exists(self.input_csv_path):
            self.logger.error(f"清洗后的数据文件不存在: {self.input_csv_path}")
            return
        
        # 读取数据并提取目标字段
        try:
            df = pd.read_csv(self.input_csv_path, encoding='utf-8-sig')
            # 检查目标字段是否存在
            for field in self.target_fields:
                if field not in df.columns:
                    self.logger.error(f"CSV中未找到「{field}」字段，请检查字段名！")
                    return
            
            self.logger.warning(f"成功读取数据，共 {len(df)} 条记录")
        except Exception as e:
            self.logger.error(f"读取CSV文件失败: {str(e)}")
            return
        
        # 对每个目标字段进行分箱处理
        for field in self.target_fields:
            self.logger.warning(f"\n===== 开始处理「{field}」字段 =====")
            # 提取并清洗数值数据（过滤非数值和异常值）
            raw_data = df[field].replace('未获取到信息', np.nan)
            numeric_data = pd.to_numeric(raw_data, errors='coerce').dropna().astype(int)
            
            if len(numeric_data) == 0:
                self.logger.warning(f"「{field}」字段无有效数值数据，跳过处理")
                continue
            
            self.logger.warning(f"有效数值数据量: {len(numeric_data)}")
            self.perform_equal_width_binning(numeric_data.values, field)
    
    def perform_equal_width_binning(self, data, field_name):
        """执行等宽分箱及各种平滑处理并打印结果"""
        # 等宽分箱（左闭右开区间）
        cuts = pd.cut(data, bins=self.bins, right=False)
        # 按区间排序统计数量
        bin_counts = pd.value_counts(cuts).sort_index()
        bin_values = [data[cuts == bin] for bin in bin_counts.index]
        
        # 构建等宽分箱矩阵
        max_count = bin_counts.max()
        width_matrix = np.full((self.bins, max_count), 0)
        for i in range(self.bins):
            bin_data = bin_values[i]
            for j in range(len(bin_data)):
                width_matrix[i, j] = bin_data[j]
        
        print(f"\n【{field_name} - 等宽分箱结果】")
        print("分箱区间:", list(bin_counts.index))
        print("分箱矩阵:")
        print(width_matrix)
        
        # 均值平滑
        mean_matrix = np.full((self.bins, max_count), 0)
        for i in range(self.bins):
            bin_data = bin_values[i]
            if len(bin_data) == 0:
                continue
            bin_mean = int(np.mean(bin_data))
            for j in range(len(bin_data)):
                mean_matrix[i, j] = bin_mean
        
        print(f"\n【{field_name} - 均值平滑结果】")
        print(mean_matrix)
        
        # 中值平滑
        median_matrix = np.full((self.bins, max_count), 0)
        for i in range(self.bins):
            bin_data = bin_values[i]
            if len(bin_data) == 0:
                continue
            bin_median = int(np.median(bin_data))
            for j in range(len(bin_data)):
                median_matrix[i, j] = bin_median
        
        print(f"\n【{field_name} - 中值平滑结果】")
        print(median_matrix)
        
        # 边界值平滑
        edge_matrix = np.full((self.bins, max_count), 0)
        for i in range(self.bins):
            bin_data = bin_values[i]
            if len(bin_data) == 0:
                continue
            left_edge = bin_data[0]
            right_edge = bin_data[-1]
            for j in range(len(bin_data)):
                if j == 0:
                    edge_matrix[i, j] = left_edge
                elif j == len(bin_data) - 1:
                    edge_matrix[i, j] = right_edge
                else:
                    # 计算到两边界的距离（用平方差比较）
                    dist_left = (bin_data[j] - left_edge) **2
                    dist_right = (bin_data[j] - right_edge)** 2
                    edge_matrix[i, j] = right_edge if dist_left > dist_right else left_edge
        
        print(f"\n【{field_name} - 边界值平滑结果】")
        print(edge_matrix)