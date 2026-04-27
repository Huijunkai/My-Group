"
 使用survival::lung数据集（肺癌患者生存数据）。
 （1） 按性别分组绘制 kaplan-Meier生存曲线， 并进行log-rank检验。
 （2） 拟合 Cox 模型， 包含年龄、性别、体力评分(ph.ecog)。
 （3）检验比例风险假设。
"

if(!require(survival)) install.packages("survival"); library(survival)
if(!require(survminer)) install.packages("survminer"); library(survminer)
data(lung)

print(lung)


#  （1） 按性别分组绘制 kaplan-Meier生存曲线， 并进行log-rank检验。
km_fit = survfit(Surv(time, status) ~ sex, data = lung)
ggsurvplot(km_fit, data = lung, pval = TRUE)
survdiff(Surv(time, status) ~ sex, data = lung)
#  （2） 拟合 Cox 模型， 包含年龄、性别、体力评分(ph.ecog)。
cox_model = coxph(Surv(time, status) ~ age + sex + ph.ecog, data = lung)
print(summary(cox_model))
#  （3）检验比例风险假设。
print(cox.zph(cox_model))
plot(cox.zph(cox_model))
