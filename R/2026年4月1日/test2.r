'
题目描述
文件夹 logs/ 下包含多个以 .log 结尾的文本文件，每个文件记录了一次用户访问日志。
文件编码为 GB18030（中文 Windows 系统常见编码）。
请完成以下任务：

（1）获取文件夹下所有 .log 文件的路径。
（2）使用 readr::read_csv() 并指定编码为 locale = locale(encoding = "GB18030") 依次读取每个文件。
（3）将所有数据框合并为一个总数据框。
（4）统计每种 action 的发生次数。
（5）将统计结果写入 CSV 文件 action_summary.csv。
'
# install.packages('readr')
# install.packages('writer')
# install.packages('dplyr')
# install.packages('purrr')

# 安装包（仅第一次运行）
# install.packages(c('readr', 'dplyr', 'purrr'))

# 加载包
library(readr)
library(dplyr)
library(purrr)

# ======================
# 【固定写法】手动设置你的脚本所在路径
# ======================
# 把这里改成你 test2.r 所在的文件夹路径
# 你的路径是：f:\\My-study\\R\\2026年4月1日
setwd("f:\\My-study\\R\\2026年4月1日")

# ======================
# 1. 获取 logs 文件夹下所有 .log 文件
# ======================
file_paths = list.files(path = "logs", pattern = "\\.log$", full.names = TRUE)

# 检查是否找到文件
if (length(file_paths) == 0) {
  stop("❌ 错误：logs 文件夹里没有 .log 文件！请检查文件夹位置！")
}

cat("✅ 找到日志文件：\n")
print(file_paths)

# ======================
# 2. 读取所有文件（GB18030 编码）
# 3. 合并成一个数据框
# ======================
all_logs = map_dfr(
  file_paths,
  ~read_csv(.x, locale = locale(encoding = "GB18030"), show_col_types = FALSE)
)

cat("\n✅ 数据读取完成：", nrow(all_logs), "行", ncol(all_logs), "列\n")
print(colnames(all_logs))  # 查看列名

# ======================
# 4. 统计 action 次数
# ======================
if (!"action" %in% colnames(all_logs)) {
  stop("❌ 数据里没有 action 这一列！请检查日志文件内容！")
}

action_summary = all_logs %>%
  group_by(action) %>%
  summarise(count = n(), .groups = "drop")

cat("\n📊 统计结果：\n")
print(action_summary)

# ======================
# 5. 保存结果
# ======================
write_csv(action_summary, "action_summary.csv")
cat("\n✅ 结果已保存成功！")