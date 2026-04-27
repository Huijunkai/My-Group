'
第2题：iris数据集——花萼长度与花瓣长度的关系及物种差异
数据：iris（R内置数据集，包含三种鸢尾花的花萼和花瓣测量数据）。
任务：
（1）对 Sepal.Length（花萼长度）进行描述性统计（均值、标准差、最小值、最大值）。
（2）按物种 Species 分组，计算 Sepal.Length 的均值和标准差，并制作柱状图（均值±误差线）。
（3）绘制 Sepal.Length 与 Petal.Length（花瓣长度）的散点图，按 Species 设置不同颜色，并添加线性回归线（全样本）。
（4）为每个物种单独添加回归线（即分别拟合三条线），并在图中显示。

'
library(ggplot2)
library(dplyr)
data(iris)
print(iris)
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

a = ggplot(iris_stat, aes(x = Species, y = mean_sepal, fill = Species)) +
      geom_col() +
      geom_errorbar(aes(ymin = mean_sepal - sd_sepal, ymax = mean_sepal +sd_sepal), width = 0.2) +
      labs(title = '各物种花萼长度（均值±标准差）', x = '物种', y = '花萼长度')
print(a)
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