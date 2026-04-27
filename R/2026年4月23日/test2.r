"
1. EuStockMarkets 数据集 – 时间序列预测
背景：EuStockMarkets 是 R 内置数据集，记录了 1991 年至 1998 年间德国 DAX、瑞士 SMI、法国 CAC、英国 FTSE 四个欧洲股票指数的每日收盘价（单位：指数点）。本题选取其中一个指数（例如 DAX）进行分析。
任务：
（1）提取 DAX 指数序列，绘制时序图，观察是否存在趋势、波动聚集等现象。判断是否需要差分或对数变换以使序列平稳。
（2）使用 auto.arima() 函数（来自 forecast 包）自动选择最优的 ARIMA 模型，可考虑季节性（交易日通常为周周期，但此处周期可设为 5 或忽略）。
（3）基于所选模型，预测未来 100 个交易日的收盘价。
（4）绘制包含历史数据与未来预测值的曲线图，并给出 95% 置信区间。
提示：数据为每日数据，可能存在非恒定方差，可尝试对数变换；auto.arima 中可设置 stepwise = FALSE 提高搜索精度。
"

# 1. EuStockMarkets 数据集 – 时间序列预测
if(!require(forecast)) install.packages("forecast"); library(forecast)
if(!require(ggplot2)) install.packages("ggplot2"); library(ggplot2)

# 加载数据集
data(EuStockMarkets)

# 提取 DAX 指数序列
dax = EuStockMarkets[, "DAX"]

# 绘制时序图
plot(dax, main = "DAX 指数时序图", xlab = "时间", ylab = "指数点")

# 观察差分后的序列
dax_diff = diff(dax)
plot(dax_diff, main = "DAX 指数一阶差分", xlab = "时间", ylab = "差分后值")

# 观察对数变换后的序列
dax_log = log(dax)
plot(dax_log, main = "DAX 指数对数变换", xlab = "时间", ylab = "对数值")

# 观察对数变换后差分的序列
dax_log_diff = diff(dax_log)
plot(dax_log_diff, main = "DAX 指数对数变换后一阶差分", xlab = "时间", ylab = "差分后值")

# 使用 auto.arima 选择最优模型
# 对对数变换后的数据进行建模
dax_log_ts = ts(dax_log, frequency = 5)  # 假设每周5个交易日
model = auto.arima(dax_log_ts, stepwise = FALSE, seasonal = TRUE)
print("最优 ARIMA 模型:")
print(model)

# 预测未来 100 个交易日
forecast_result = forecast(model, h = 100)

# 绘制预测结果
plot(forecast_result, main = "DAX 指数预测", xlab = "时间", ylab = "对数值")

# 转换回原始尺度
forecast_original = exp(forecast_result$mean)
lower_original = exp(forecast_result$lower)
upper_original = exp(forecast_result$upper)

# 绘制原始尺度的预测结果
plot(dax, main = "DAX 指数预测（原始尺度）", xlab = "时间", ylab = "指数点")
lines(forecast_original, col = "red", lwd = 2)
lines(lower_original[, "95%"], col = "blue", lty = 2)
lines(upper_original[, "95%"], col = "blue", lty = 2)
legend("topleft", legend = c("历史数据", "预测值", "95% 置信区间"), col = c("black", "red", "blue"), lty = c(1, 1, 2), lwd = c(1, 2, 1))


"
2. ovarian 数据集 – 生存分析
背景：ovarian 是 survival 包中的数据集，记录了 26 例卵巢癌患者的生存数据。变量包括：futime（生存时间，天）、fustat（生存状态，1=死亡，0=删失）、age（年龄）、rx（治疗组别，1=单药，2=联合用药）、resid.ds（残留病灶程度，1=无，2=有）、ecog.ps（ECOG 体力评分）。
任务：
（1）按治疗组别（rx）分组，绘制 Kaplan-Meier 生存曲线，并添加风险表格（可选）。
（2）进行 log-rank 检验，比较两组生存曲线是否存在显著差异。
（3）拟合 Cox 比例风险模型，以 futime 和 fustat 为生存结局，协变量包括：age、rx、resid.ds。
（4）使用 cox.zph() 检验比例风险假设，并解释结果（是否满足假设，若不满足如何调整）。
提示：使用 survival 包的 Surv() 和 coxph() 函数；survminer 包可增强绘图。
"

# 2. ovarian 数据集 – 生存分析
if(!require(survival)) install.packages("survival"); library(survival)
if(!require(survminer)) install.packages("survminer"); library(survminer)

# 加载数据集
data(ovarian)

print("ovarian 数据集:")
print(ovarian)

# （1）按治疗组别（rx）分组，绘制 Kaplan-Meier 生存曲线
km_fit = survfit(Surv(futime, fustat) ~ rx, data = ovarian)
ggsurvplot(km_fit, data = ovarian, pval = TRUE, risk.table = TRUE, main = "不同治疗组的 Kaplan-Meier 生存曲线")

# （2）进行 log-rank 检验
surv_diff = survdiff(Surv(futime, fustat) ~ rx, data = ovarian)
print("Log-rank 检验结果:")
print(surv_diff)

# （3）拟合 Cox 比例风险模型
cox_model = coxph(Surv(futime, fustat) ~ age + rx + resid.ds, data = ovarian)
print("Cox 比例风险模型结果:")
print(summary(cox_model))

# （4）检验比例风险假设
zph_result = cox.zph(cox_model)
print("比例风险假设检验结果:")
print(zph_result)
plot(zph_result, main = "比例风险假设检验")