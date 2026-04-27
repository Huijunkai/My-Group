# 创建数据库sales date（7天），product （a，b，c，每天每种产品至少有一条记录）
# quantity（1-20随机）
# price（单价 a：10，b：15，c：20）

# 安装依赖包
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")

# 加载dplyr包
library(dplyr)

# 使用expand.grid函数生成数据（推荐写法）
dates = seq.Date(as.Date('2023-01-01'), as.Date('2023-01-07'), by = "days")
products = c('A', 'B', 'C')
product_price_map = c(A = 10, B = 15, C = 20)

sales = expand.grid(date = dates, product = products) %>%
  mutate(quantity = sample(1:20, n(), replace = TRUE),
         price = product_price_map[product])

print(sales)

# (1) 添加一列revenue，计算每种产品的销售额（quantity * price）
sales1 = sales %>% 
  mutate(revenue = quantity * price)
print(sales1)

#（2）计算每种产品在一周内的总销售额和总销售量，并按照总销售额度降序排序。
sales_summary = sales1 %>%
  group_by(product) %>%
  summarise(total_revenue = sum(revenue), total_quantity = sum(quantity)) %>%
  arrange(desc(total_revenue))
print(sales_summary)

# （3）找出那一天的总销售额最高，输出该日期和总销售额。
daily_revenue = sales1 %>%
  group_by(date)%>%
  summarise(daily_revenue = sum(revenue)) %>%
  arrange(desc(daily_revenue))%>%
  head(1)
print(daily_revenue)

# （4）筛选处销售量超过10的记录，并按照日期升序、产品名升序排列输出
filtered_sales = sales1 %>%
  filter(quantity > 10) %>%
  arrange(date, product)
print(filtered_sales)