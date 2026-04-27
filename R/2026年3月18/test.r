# x = c(1,2,3)
# m = matrix(x, nrow = 1, ncol = 3)
# arr = array(m, dim = 3,dimnames =  list(c('x','y','z')))
# arr[1] = NA
# print(arr)

# s = c('男','女','男','女')
# se = factor(s,ordered = TRUE, nmax = 2)

# se[1] = '女'
# print(se)

table = data.frame(
    "姓名" = c("张三", "李四", "王五"),
    "年龄" = c(20, 25, 30),
    "性别" = c("男", "女", "男"),
    "学号" = c(1001, 1002, 1003),
    "月薪" = c(5000, 6000, 7000)
)
print(table)

summary(table)

c1 = c(1, 2, 3)
c2 = c(1, 1, 1)
c3 = c(3, 4)

df = cbind(c1, c2, c3)
print(is.data.frame(df))