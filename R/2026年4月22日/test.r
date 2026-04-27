'
使用内置数据集 warpbreaks （羊毛和战力水平对织布机断裂次数的影响）。
(1)建立泊松回归，以wool和tension为自动变量。
(2)检验是否存在过离散。
(3)若过离散，改用负二项回归(MASS::glm.nb)。

'


# 加载数据
data(warpbreaks)
print(warpbreaks)

# (1)建立泊松回归，以wool和tension为自动变量。
pois_model = glm(breaks ~ wool + tension, data = warpbreaks, family = poisson())
print(summary(pois_model))

# (2)检验是否存在过离散。
if(!require(AER)) install.packages('AER'); library(AER)

dispersiontest(pois_model)

# (3)若过离散，改用负二项回归(MASS::glm.nb)。
if(!require(MASS)) install.packages('MASS'); library(MASS)

nb_model = glm.nb(breaks ~ wool + tension, data = warpbreaks)
print(summary(nb_model))