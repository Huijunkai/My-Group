'1. 线性回归练习题：基于 swiss 数据集的逐步回归分析‌
使用 R 语言中的 swiss 数据集（包含瑞士47个省份的社会经济指标），完成以下任务：
（1）以 Fertility 为因变量，其余变量（Agriculture, Examination, Education, Catholic, Infant.Mortality）为自变量，建立全模型。
（2）使用 step() 函数进行‌逐步回归‌，设定 direction = "both"，基于 AIC 准则自动选择最优变量子集。
（3）输出最终模型的回归方程，并解释所保留变量的实际意义。
'
# 加载数据集
data(swiss)

# 查看数据集结构
str(swiss)

# （1）建立全模型
full_model = lm(Fertility ~ Agriculture + Examination + Education + Catholic + Infant.Mortality, data = swiss)
summary(full_model)

# （2）使用 step() 函数进行逐步回归，基于 AIC 准则
step_model = step(full_model, direction = "both")

# 查看最终模型
print(summary(step_model))

# （3）输出最终模型的回归方程
cat("\n最终回归方程：\n")
cat("Fertility = ", round(coefficients(step_model)[1], 4), " ",
    ifelse(coefficients(step_model)[2] >= 0, "+", ""), round(coefficients(step_model)[2], 4), " * Agriculture ",
    ifelse(coefficients(step_model)[3] >= 0, "+", ""), round(coefficients(step_model)[3], 4), " * Examination ",
    ifelse(coefficients(step_model)[4] >= 0, "+", ""), round(coefficients(step_model)[4], 4), " * Education ",
    ifelse(coefficients(step_model)[5] >= 0, "+", ""), round(coefficients(step_model)[5], 4), " * Catholic ",
    ifelse(coefficients(step_model)[6] >= 0, "+", ""), round(coefficients(step_model)[6], 4), " * Infant.Mortality\n",
    sep = "")

# 解释所保留变量的实际意义
cat("\n变量解释：\n")
cat("- Agriculture：农业劳动力占比，系数为正，说明农业占比越高，生育率越高\n")
cat("- Examination：体检率，系数为负，说明体检率越高，生育率越低\n")
cat("- Education：教育水平，系数为负，说明教育水平越高，生育率越低\n")
cat("- Catholic：天主教徒比例，系数为正，说明天主教徒比例越高，生育率越高\n")
cat("- Infant.Mortality：婴儿死亡率，系数为正，说明婴儿死亡率越高，生育率越高\n")

