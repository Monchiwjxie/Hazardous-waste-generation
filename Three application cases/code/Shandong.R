# R包导入 --------------------------------------------------------------------
library(mlr3verse)
library(tidyverse)
library(caret)
library(mltools)
library(data.table)
library(UBL)
library(progress)
# 模型构建 --------------------------------------------------------------------
#### 数据导入 ####
data <- read.csv("Model_SD_update.csv") %>% 
  select(-Code, -Sector) 
names(data) <- c(
  "process_1", "process_2","process_3","process_4","process_5","process_6",
  "Size",
  "Wastewater", "COD","NH3N","TN",
  "TP","Cr","Cr6",
  "HWsum"
)
data <- data %>% 
  mutate( process_1 = as.factor(process_1),
          process_2 = as.factor(process_2),
          process_3 = as.factor(process_3),
          process_4 = as.factor(process_4),
          process_5 = as.factor(process_5),
          process_6 = as.factor(process_6),
          Size = as.factor(Size)
  )

set.seed(759)
sample = sample(length(data$HWsum), 0.2*length(data$HWsum))
data_test = data[sample,]
data_train = data[-sample,]

write.csv(data_test, "山东省_test.csv")
write.csv(data_train_smogn, "山东省_train_smogn.csv")
#### 数据平衡 #### 
set.seed(1234)
data_train_smogn = SMOGNRegress(form = HWsum ~ .,
                                dat = data_train,
                                thr.rel = 0.24,
                                dist = "HEOM",
                                k = 5,
                                C.perc = "balance")

#### 创建任务 ####
# task = as_task_regr(data_train, target = "HWsum")
task_smogn = as_task_regr(data_train_smogn, target = "HWsum")
# Ranger_model = lrn("regr.ranger",
#                    num.trees = 500,
#                    min.node.size = 2,
#                    mtry = 3)

Ranger_model_smogn = lrn("regr.ranger",
                         num.trees = 500,
                         min.node.size = 2,
                         mtry = 3)

Ranger_model_smogn$train(task_smogn, row_ids = 1:length(data_train_smogn$HWsum))
#### 测试集计算 ####
# test_result = Ranger_model$predict_newdata(data_test)
test_result_smogn = Ranger_model_smogn$predict_newdata(data_test)
# 
# r2 = test_result$score(msr("regr.rsq"))
# r2_smogn = test_result_smogn$score(msr("regr.rsq"))

result_rsq = test_result_smogn$score(msr("regr.rsq"))
result_rmse = test_result_smogn$score(msr("regr.rmse"))
result_mae = test_result_smogn$score(msr("regr.mae"))
result_mape = test_result_smogn$score(msr("regr.mape"))
result_sse = test_result_smogn$score(msr("regr.sse"))

# 
#### 测试集真实值-预测值结果导出 ####
test_truth = as.data.frame(test_result_smogn$truth)
test_response = as.data.frame(test_result_smogn$response)
truth_response = cbind(test_truth, test_response)
mad(truth_response$`test_result_smogn$truth`, truth_response$`test_result_smogn$response`)
