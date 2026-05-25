library(ggplot2)
library(dplyr)

a = head(mtcars)
# print(a)

# 画图
b = ggplot(mtcars, aes(x = wt, y = mpg)) + 
  geom_point(size = 3, color = "steelblue") +
  labs(title = "汽车重量与油耗关系",
       x = "车重（1000磅）",
       y = "油耗（英里/加仑）") +
  theme_minimal()

print(b)

# (1) 对mpg（英里/加仑）变量，计算均值、中位数、标准差、四分数
mtcars %>% summarise(
    mean_mpg = mean(mpg),
    median_mpg = median(mpg),
    std_mpg = sd(mpg),
    q1_mpg = quantile(mpg, 0.25),
    q3_mpg = quantile(mpg, 0.75)
)%>%
    print()

# (2)绘制mpg的直方图，设置合适的组数，添加密度曲线
c = ggplot(mtcars, aes(x = mpg)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 10,
    fill = 'lightblue',
    col = 'black'
  ) + 
  geom_density(col = 'red', linewidth = 1) +
  labs(
    title = 'mpg的直方图',
    x = 'Miles per Gallon',
    y = 'Density'
  )

print(c)

# (3)绘制mpg的箱线图，观察离群点
d = ggplot(mtcars ,aes(y = mpg)) +
        geom_boxplot(fill = 'orange') +
        labs(title = 'mpg的箱线图', y = 'mpg')
print(d)

# (4)按气缸数 cyl(4,6,8)分组，绘制mpg的箱线图
e = ggplot(mtcars,aes(x = factor(cyl),y = mpg, fill = factor(cyl))) +
        geom_boxplot() + 
        labs(title = '不同气缸数的mpg的箱线图', x = 'Cylinders' ,y = 'mpg') +
        theme_minimal()
print(e)

'
第2题：iris数据集——花萼长度与花瓣长度的关系及物种差异
数据：iris（R内置数据集，包含三种鸢尾花的花萼和花瓣测量数据）。
任务：
（1）对 Sepal.Length（花萼长度）进行描述性统计（均值、标准差、最小值、最大值）。
（2）按物种 Species 分组，计算 Sepal.Length 的均值和标准差，并制作柱状图（均值±误差线）。
（3）绘制 Sepal.Length 与 Petal.Length（花瓣长度）的散点图，按 Species 设置不同颜色，并添加线性回归线（全样本）。
（4）为每个物种单独添加回归线（即分别拟合三条线），并在图中显示。

'

data(iris)

# (1) 对 Sepal.Length 描述性统计
iris %>% summarise(
  mean_iris = mean(Sepal.Length),
  std_iris = sd(Sepal.Length),
  min_iris = min(Sepal.Length),
  max_iris = max(Sepal.Length)
) %>% 
print()

# (2) 按Species分组计算均值、标准差 + 柱状图（误差线）
iris_stat = iris %>% 
  group_by(Species) %>% 
  summarise(
    mean_sepal = mean(Sepal.Length),
    sd_sepal = sd(Sepal.Length)
  )

print(iris_stat)

p2 = ggplot(iris_stat, aes(x = Species, y = mean_sepal, fill = Species)) +
  geom_col() +
  geom_errorbar(aes(ymin = mean_sepal - sd_sepal, ymax = mean_sepal + sd_sepal), width = 0.2) +
  labs(title = "各物种花萼长度（均值±标准差）", x = "物种", y = "花萼长度") +
  theme_minimal()

print(p2)

# (3) 散点图 + 全样本回归
p3 = ggplot(iris, aes(x = Sepal.Length, y = Petal.Length)) +
  geom_point(aes(color = Species), size = 2) +
  geom_smooth(method = "lm", se = F, color = "black") +
  labs(title = "花萼长度与花瓣长度（全样本回归）", x = "花萼长度", y = "花瓣长度") +
  theme_minimal()

print(p3)

# (4) 散点图 + 每个物种单独回归线
p4 = ggplot(iris, aes(x = Sepal.Length, y = Petal.Length, color = Species, fill = Species)) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", se = T) +
  labs(title = "花萼长度与花瓣长度（分物种回归）", x = "花萼长度", y = "花瓣长度") +
  theme_minimal()

print(p4)