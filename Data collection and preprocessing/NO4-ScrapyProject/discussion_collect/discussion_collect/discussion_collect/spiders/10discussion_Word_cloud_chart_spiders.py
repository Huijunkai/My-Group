""" 
此子项目用于实现词云图的设计，词云图设计是一种数据可视化的技术，使用python 而非findebi工具。
对清洗后的详细页内容中的正文和评论（英文内容）分别进行词云生成，提取各自的十大热点词汇，生成组合式词云图片，
图片标题用中文描述，帮助作者了解正文核心主题和评论舆论方向，以便后续发布的内容进行优化和改进。
"""
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scrapy import Spider
from wordcloud import WordCloud  # 导入词云库

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 非名词过滤表（覆盖代词/副词/介词/连词/助动词等）
NON_NOUNS = {
    # 代词/限定词
    'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
    'it', 'its', 'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs', 'this', 'that',
    'these', 'those', 'all', 'any', 'both', 'each', 'few', 'many', 'much', 'none', 'some', 'such',
    'a', 'an', 'the', 'every', 'no', 'neither', 'either', 'other', 'another',
    # 副词
    'here', 'there', 'when', 'where', 'why', 'how', 'now', 'then', 'soon', 'often', 'always',
    'never', 'maybe', 'perhaps', 'quite', 'very', 'too', 'so', 'just', 'almost', 'already',
    # 介词/连词
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'out', 'off', 'over',
    'under', 'and', 'or', 'but', 'if', 'because', 'as', 'while', 'until', 'since', 'though',
    # 助动词/情态动词
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
    # 业务无效词
    '未获取到信息', 'request', 'failed', 'parse', 'error', 'content', 'comment', 'unknown', 'user'
}

# 2. 名词特征辅助筛选（排除动词/形容词后缀）
VERB_ADJ_SUFFIX = ['ing', 'ed', 'en', 'able', 'ible', 'ful', 'less', 'ive', 'ous', 'ly']

class DiscussionSpider(Spider):
    name = 'discussion_Word_cloud_chart_spiders'
    
    # 路径配置（和之前一致）
    input_csv_path = os.path.join('data', 'csv', 'discussion_Detailed_Page_cleaned.csv')
    output_img_path = os.path.join('data', 'images', 'content_comment_nouns_top50_en.png')  # 文件名更新为top50
    comment_separator = '|||'
    
    def start_requests(self):
        """核心执行逻辑：读取CSV→过滤名词→生成词云图"""
        # 检查输入文件
        if not os.path.exists(self.input_csv_path):
            self.logger.error(f"❌ 输入CSV不存在：{self.input_csv_path}")
            return
        
        # 读取CSV
        try:
            df = pd.read_csv(
                self.input_csv_path,
                encoding='utf-8-sig',
                usecols=['清洗后的正文内容', '清洗后的评论内容列表']
            )
            self.logger.info(f"✅ 读取到 {len(df)} 条有效记录")
        except Exception as e:
            self.logger.error(f"❌ 读取CSV失败：{str(e)}")
            return
        
        # 处理文本+筛选名词
        content_nouns = self._extract_nouns(df['清洗后的正文内容'], is_content=True)
        comment_nouns = self._extract_nouns(df['清洗后的评论内容列表'], is_content=False)
        
        # 生成纯名词词云图
        self._generate_wordcloud(content_nouns, comment_nouns)
        
        # 输出Top50名词
        self._print_top50_nouns(content_nouns, comment_nouns)  # 方法名更新为top50
        
        self.logger.info(f"✅ 纯名词词云图已保存：{os.path.abspath(self.output_img_path)}")
        return
    
    def _extract_nouns(self, text_series, is_content):
        """提取文本中的纯名词（无nltk版）"""
        # 过滤无效值
        invalid = ['', ' ', '未获取到正文内容', '未获取到评论内容', '请求失败', '解析失败',
                   'No content', 'No comments', 'Request failed', 'Parse error']
        valid_text = text_series.dropna().loc[~text_series.isin(invalid)]
        
        # 合并文本
        if not is_content:
            # 处理评论：拆分列表+移除用户名前缀
            clean_comments = []
            for comment_str in valid_text:
                for c in comment_str.split(self.comment_separator):
                    clean_c = re.sub(r'^[^:]+:\s*', '', c).strip()
                    clean_comments.append(clean_c)
            combined_text = ' '.join(clean_comments)
        else:
            combined_text = ' '.join(valid_text.tolist())
        
        # 清洗文本：小写+移除非字母+拆分单词
        combined_text = combined_text.lower()
        combined_text = re.sub(r'[^a-zA-Z\s]', ' ', combined_text)
        words = combined_text.split()
        
        # 筛选名词：核心逻辑
        nouns = []
        for word in words:
            # 条件1：排除非名词表中的词
            if word in NON_NOUNS:
                continue
            # 条件2：长度≥2（排除单字母）
            if len(word) < 2:
                continue
            # 条件3：排除动词/形容词后缀
            if any(word.endswith(suffix) for suffix in VERB_ADJ_SUFFIX):
                continue
            # 条件4：排除纯数字
            if word.isdigit():
                continue
            # 符合所有条件 → 判定为名词
            nouns.append(word)
        
        # 统计名词词频
        return dict(Counter(nouns))
    
    def _generate_wordcloud(self, content_nouns, comment_nouns):
        """生成纯名词Top50组合式词云图"""
        # 创建输出目录
        os.makedirs(os.path.dirname(self.output_img_path), exist_ok=True)
        
        # 提取Top50名词及频次（关键修改：10→50）
        content_top50 = Counter(content_nouns).most_common(50)
        comment_top50 = Counter(comment_nouns).most_common(50)
        
        # 画布设置（保持1行2列的组合布局，适当放大画布适配更多词汇）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))  # 画布放大：16,8 → 20,10
        fig.suptitle('正文VS评论Top50纯名词词云（英文）', fontsize=20, fontweight='bold', y=0.95)
        
        # --------------------- 正文词云生成 ---------------------
        if content_top50:
            # 将Top50转换为词云可识别的字典格式
            content_word_freq = dict(content_top50)
            # 配置词云样式（max_words改为50，适配更多词汇）
            content_wc = WordCloud(
                background_color='white',
                width=800, height=600,  # 词云尺寸放大：600,500 → 800,600
                max_words=50,  # 关键修改：10→50
                relative_scaling=0.6,  # 降低关联度，避免部分词汇过大
                random_state=42  # 固定随机种子，保证样式一致
            ).generate_from_frequencies(content_word_freq)
            # 在子图1中显示词云
            ax1.imshow(content_wc, interpolation='bilinear')
            ax1.set_title('正文Top50纯名词', fontsize=16, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, '无有效名词数据', ha='center', va='center', fontsize=14)
            ax1.set_title('正文Top50纯名词', fontsize=16, fontweight='bold')
        ax1.axis('off')  # 隐藏坐标轴
        
        # --------------------- 评论词云生成 ---------------------
        if comment_top50:
            # 将Top50转换为词云可识别的字典格式
            comment_word_freq = dict(comment_top50)
            # 配置词云样式
            comment_wc = WordCloud(
                background_color='whitesmoke',
                width=800, height=600,
                max_words=50,  # 关键修改：10→50
                relative_scaling=0.6,
                random_state=42
            ).generate_from_frequencies(comment_word_freq)
            # 在子图2中显示词云
            ax2.imshow(comment_wc, interpolation='bilinear')
            ax2.set_title('评论Top50纯名词', fontsize=16, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, '无有效名词数据', ha='center', va='center', fontsize=14)
            ax2.set_title('评论Top50纯名词', fontsize=16, fontweight='bold')
        ax2.axis('off')  # 隐藏坐标轴
        
        # 保存图片（高分辨率，适配中文标题）
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(self.output_img_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _print_top50_nouns(self, content_nouns, comment_nouns):
        """日志输出Top50纯名词（关键修改：10→50）"""
        self.logger.info("\n===== 正文Top50纯名词（英文） =====")
        content_top50 = Counter(content_nouns).most_common(50)
        if content_top50:
            for idx, (word, count) in enumerate(content_top50, 1):
                self.logger.info(f"{idx}. {word} - {count}次")
        else:
            self.logger.info("无有效正文名词")
        
        self.logger.info("\n===== 评论Top50纯名词（英文） =====")
        comment_top50 = Counter(comment_nouns).most_common(50)
        if comment_top50:
            for idx, (word, count) in enumerate(comment_top50, 1):
                self.logger.info(f"{idx}. {word} - {count}次")
        else:
            self.logger.info("无有效评论名词")