'1. ChickWeight 数据集 – 线性混合模型
背景：ChickWeight 数据集记录了不同饮食对小鸡体重增长的影响。每只小鸡在不同时间点（0, 2, 4, …, 20 天）多次测量体重。数据包含变量：weight（体重，克）、Time（时间，天）、Chick（小鸡编号，1~50）、Diet（饮食类型，1~4）。
任务：
（1）建立第一个线性混合模型：以 weight 为响应变量，Time 为固定效应（线性趋势），Chick 为随机截距（即允许每只小鸡的初始体重不同）。
（2）建立第二个线性混合模型：在随机截距的基础上，再添加随机斜率，允许每只小鸡的 Time 效应（生长速度）也随个体变化。
（3）使用似然比检验（LRT）或信息准则（AIC/BIC）比较两个模型，并解释哪个模型更优，以及随机斜率的意义。
提示：可使用 lme4 包中的 lmer() 函数，模型表达式分别为 weight ~ Time + (1 | Chick) 和 weight ~ Time + (Time | Chick)。
'

# 1. ChickWeight 数据集分析
if(!require(lme4)) install.packages('lme4'); library(lme4)

# 加载数据集
data(ChickWeight)

# 查看数据结构
str(ChickWeight)

# 建立第一个模型：随机截距模型
model1 = lmer(weight ~ Time + (1 | Chick), data = ChickWeight)
summary(model1)

# 建立第二个模型：随机截距+斜率模型
model2 = lmer(weight ~ Time + (Time | Chick), data = ChickWeight)
summary(model2)

# 2. 模型比较
# 信息准则比较
AIC(model1, model2)
BIC(model1, model2)

# 似然比检验
anova(model1, model2)


'2. USAccDeaths 数据集 – ARIMA 建模与预测
背景：USAccDeaths 是 R 内置数据集，记录了 1973 年 1 月至 1978 年 12 月美国每月意外死亡人数（单位：人）。该序列通常表现出趋势和季节性。
任务：
（1）绘制原始时序图，观察是否存在趋势、季节性以及方差是否稳定。根据图形判断是否需要进行差分或数据变换（如对数变换）。
（2）使用 auto.arima() 函数（来自 forecast 包）自动选择最优的 ARIMA 模型，包括非季节性和季节性差分阶数。
（3）基于所选模型，预测未来 24 个月（即 1979 年 1 月至 1980 年 12 月）的意外死亡人数。
（4）绘制包含历史数据和未来预测值的曲线图，并给出 95% 置信区间。
提示：由于数据为月度序列，季节周期应为 12。auto.arima() 中需设置 seasonal = TRUE。
'

# 2. USAccDeaths 数据集分析
if(!require(forecast)) install.packages('forecast'); library(forecast)

# 加载数据集
data(USAccDeaths)

# （1）绘制原始时序图
plot(USAccDeaths, main = "USAccDeaths 原始时序图", xlab = "时间", ylab = "死亡人数")

# （2）自动选择最优 ARIMA 模型
model = auto.arima(USAccDeaths, seasonal = TRUE)
summary(model)

# （3）预测未来 24 个月
forecast_result = forecast(model, h = 24)

# （4）绘制预测图
plot(forecast_result, main = "USAccDeaths 预测图", xlab = "时间", ylab = "死亡人数")

# 显示预测结果
print(forecast_result)