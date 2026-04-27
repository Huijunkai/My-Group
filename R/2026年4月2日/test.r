'
常用的绘图函数
'
# install.packages('ggplot2')

library(ggplot2)

# 1.绘制散点图，使用plot()
x = -20:20
y = x^2

plot(x,
        y,
        pch=16,#设置形状
        main='散点图',
        xlab= 'X轴',
        ylab= 'Y轴',
        col='red'
    )

# 2.绘制折线图，使用lines()
lines(x,
        y,      
        col='blue',
        lwd = 3 #线条粗细
    )

# 设置图例,使用legend()
legend('topleft',
        legend = c('数据点','拟合线'),
        col = c('red','blue'),
        pch = c(16, NA),
        lwd = c(NA,3)
    )

# 3.绘制饼图，使用pie()
data = c(2000,2000,5000,6000,7000)
names= c('张三','李四','王五','赵六','钱八')
pie(data,
    labels =  names,
    main = '工资图表'
    )

# 4.绘制条形图，也就是柱状图，使用barplot()
barplot(data,
        main = '工资图表',
        col= c('green','red','blue','yellow','black'),
        # names= c('张三','李四','王五','赵六','钱八')
        names = names
        )

# 5.绘制箱线图，使用boxplot()
boxplot(data,
        main = '工资图表'
        )

# 6.绘制直方图，使用hist()
hist(data,
    xlab = '金额',
    ylab = '次数',
    xlim = c(2000,5000),
    ylim = c(1,3),
    las = 1,
    main = '工资图表'
    )

# 7.绘制函数曲线，使用curve()
# curve(sin(x), from = -100, to = 100)


# 外部
# ggplot(data = 数据框, aes(x = 变量, y = 变量))
