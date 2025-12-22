"""
此子项目的主要内容是对清洗后的文件中的数值进行等深分箱处理，
并对分箱结果进行均值/中值/边界值平滑处理，
以便后续的数据分析与建模使用。 
"""
from scrapy import Spider
import numpy as np
import math
import pandas as pd
import os

class DiscussionSpider(Spider):
    name = 'discussion_Equal-depth-binning_spiders'
    
    # 配置项（与等宽分箱格式对齐）
    # Clean后的数据文件路径
    input_csv_path = os.path.join('data', 'csv', 'fedora_centos_topics_cleaned.csv')
    # 目标数值字段（需分箱的字段）
    target_fields = ['总评论数', '阅读量']
    # 等深分箱配置：每个分箱的样本数量（或分箱个数，二选一）
    bin_depth = 20  # 每个分箱4条数据（示例值，可根据数据量调整）
    # 无效值过滤列表
    invalid_values = ['未获取到信息', 'N/A', '', ' ', 'NaN', 'None']

    def start_requests(self):
        """Scrapy入口：启动等深分箱处理流程"""
        self.logger.info("===== 开始执行等深分箱处理 =====")
        
        # 1. 加载并预处理Clean后的数值数据
        field_data_dict = self.load_and_preprocess_data()
        if not field_data_dict:
            self.logger.error("数据加载失败，终止等深分箱流程")
            return
        
        # 2. 对每个字段执行等深分箱+平滑处理
        for field_name, numeric_data in field_data_dict.items():
            self.logger.info(f"\n----- 处理字段：{field_name} -----")
            if len(numeric_data) == 0:
                self.logger.info(f"字段「{field_name}」无有效数值数据，跳过")
                continue
            
            # 执行等深分箱核心逻辑
            self.perform_equal_depth_binning(numeric_data, field_name)
        
        self.logger.info("\n===== 等深分箱处理全部完成 =====")
        return  # 无网络请求，直接返回

    def load_and_preprocess_data(self):
        """加载Clean后的CSV，预处理数值字段（过滤无效值+转数值型）"""
        # 检查文件是否存在
        if not os.path.exists(self.input_csv_path):
            self.logger.error(f"Clean文件不存在：{self.input_csv_path}")
            return {}
        
        # 读取CSV并处理
        try:
            df = pd.read_csv(
                self.input_csv_path,
                encoding='utf-8-sig',
                dtype=str  # 先读为字符串，避免自动类型错误
            )
            self.logger.info(f"成功读取Clean数据，共 {len(df)} 条记录")
            
            # 检查目标字段是否存在
            missing_fields = [f for f in self.target_fields if f not in df.columns]
            if missing_fields:
                self.logger.error(f"CSV缺失目标字段：{missing_fields}")
                return {}
            
            # 预处理每个数值字段
            field_data_dict = {}
            for field in self.target_fields:
                # 过滤无效值 + 转数值型
                clean_series = df[field].replace(self.invalid_values, np.nan)
                numeric_series = pd.to_numeric(clean_series, errors='coerce').dropna()
                field_data = numeric_series.values.astype(float)  # 统一转为float避免类型问题
                field_data_dict[field] = field_data
                self.logger.info(f"字段「{field}」有效数值量：{len(field_data)}")
            
            return field_data_dict
        
        except Exception as e:
            self.logger.error(f"加载数据失败：{str(e)}")
            return {}

    def perform_equal_depth_binning(self, data, field_name):
        """执行等深分箱核心逻辑 + 均值/中值/边界值平滑"""
        # 步骤1：数据排序（等深分箱前提）
        sorted_data = np.sort(data)
        total_samples = len(sorted_data)
        
        # 步骤2：计算分箱数（确保每个分箱样本数=bin_depth，最后一个分箱补全）
        bin_count = math.ceil(total_samples / self.bin_depth)
        # 补齐数据（使总样本数为bin_depth的整数倍，避免分箱不均）
        pad_length = (bin_count * self.bin_depth) - total_samples
        if pad_length > 0:
            padded_data = np.pad(sorted_data, (0, pad_length), mode='edge')  # 用边缘值补齐
        else:
            padded_data = sorted_data
        
        # 步骤3：等深分箱（按深度拆分）
        depth_bins = padded_data.reshape(bin_count, self.bin_depth)
        self.logger.info(f"等深分箱完成，分箱数：{bin_count}，每个分箱样本数：{self.bin_depth}")
        
        # 打印原始等深分箱结果
        print(f"\n【{field_name} - 等深分箱原始结果】")
        print(depth_bins)
        
        # 步骤4：均值平滑
        mean_depth = np.full(depth_bins.shape, 0.0)
        for i in range(depth_bins.shape[0]):
            bin_mean = np.mean(depth_bins[i])
            mean_depth[i] = bin_mean  # 整行赋值，简化循环
        print(f"\n【{field_name} - 等深分箱·均值平滑结果】")
        print(mean_depth)
        
        # 步骤5：中值平滑
        median_depth = np.full(depth_bins.shape, 0.0)
        for i in range(depth_bins.shape[0]):
            bin_median = np.median(depth_bins[i])
            median_depth[i] = bin_median
        print(f"\n【{field_name} - 等深分箱·中值平滑结果】")
        print(median_depth)
        
        # 步骤6：边界值平滑（左/右边界，距离近的赋值）
        edge_depth = np.full(depth_bins.shape, 0.0)
        edge_left = depth_bins[:, 0]  # 每个分箱的左边界（第一列）
        edge_right = depth_bins[:, -1]  # 每个分箱的右边界（最后一列）
        
        for i in range(depth_bins.shape[0]):
            for j in range(depth_bins.shape[1]):
                if j == 0:  # 第一列=左边界
                    edge_depth[i][j] = edge_left[i]
                elif j == self.bin_depth - 1:  # 最后一列=右边界
                    edge_depth[i][j] = edge_right[i]
                else:  # 中间值：判断距离左/右边界更近
                    dist_left = math.pow((edge_left[i] - depth_bins[i][j]), 2)
                    dist_right = math.pow((edge_right[i] - depth_bins[i][j]), 2)
                    edge_depth[i][j] = edge_right[i] if dist_left > dist_right else edge_left[i]
        
        print(f"\n【{field_name} - 等深分箱·边界值平滑结果】")
        print(edge_depth)

        # 补充分箱统计日志
        self.logger.info(f"字段「{field_name}」分箱统计：")
        self.logger.info(f"  - 原始数据量：{len(data)}")
        self.logger.info(f"  - 补齐后数据量：{len(padded_data)}")
        self.logger.info(f"  - 分箱数量：{bin_count}")
        self.logger.info(f"  - 每个分箱样本数：{self.bin_depth}")