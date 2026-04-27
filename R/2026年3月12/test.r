# x = '123456334.326222'
# f = format(x, digits = 3, nsmall = 2,scientific = FALSE,width = 30, justify = "left")
# print(f)

# s= 'hello world!'
# print(substring(s, 3, 6))


list1 = list(1,2,3,4,5)
# print(list1)
list1[4] = NULL
print(list1)

data = c(1,2,3,4,5,6,7,8)
m = outMatrix <- matrix(data = data, nrow = 2, ncol = 4)
print(m)

data2 = c(1,1,1,1,1,1)
m2 = matrix(data = data2, nrow = 4, ncol = 2)
print(m2)
print(m * m)
print(m %*% m2)