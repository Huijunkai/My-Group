'
 使用AirPassengers 数据集（1949-1960年每月国际航空乘客数）。
 （1） 绘制时序图，判断是否需要差分。
 （2） 使用auto.arima 自动选择最佳模型。
 （3） 预测未来24个月，并绘制预测图
'
if(!require(forecast)) install.packages('forecast'); library(forecast)


# 加载数据
data(AirPassengers)
print(AirPassengers)

#  （1） 绘制时序图，判断是否需要差分。
print(autoplot(AirPassengers, main = '时序图', col = 'red'))

#  （2） 使用auto.arima 自动选择最佳模型。
fit = auto.arima(AirPassengers, seasonal = TRUE)
print(summary(fit))

#  （3） 预测未来24个月，并绘制预测图
forecast = forecast(fit, h = 2400)
print(autoplot(forecast, main = '未来24个月的预测图', col = 'red'))