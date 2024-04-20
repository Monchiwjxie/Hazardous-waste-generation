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
data <- read.csv("Model_ZJ_update.csv")
names(data) <- c(
  "process_1", "process_2","process_3","process_4","process_5","process_6",
  "Size",
  "Wastewater", "COD","NH3N","TN",
  "TP","Cr","Cr6",
  "HWsum"
)

set.seed(529)
sample = sample(length(data$HWsum), 0.2*length(data$HWsum))
data_test = data[sample,]
data_train = data[-sample,]

#### 数据平衡 #### 
set.seed(1234)
data_train_smogn = SMOGNRegress(form = HWsum ~ .,
                                dat = data_train,
                                thr.rel = 0.24,
                                dist = "HEOM",
                                k = 5,
                                C.perc = "balance")

#### 创建任务 ####
task_smogn = as_task_regr(data_train_smogn, target = "HWsum")

Ranger_model_smogn = lrn("regr.ranger",
                         num.trees = 630,
                         min.node.size = 2,
                         mtry = 3)

Ranger_model_smogn$train(task_smogn, row_ids = 1:length(data_train_smogn$HWsum))
#### 测试集计算 ####
# test_result = Ranger_model$predict_newdata(data_test)
test_result_smogn = Ranger_model_smogn$predict_newdata(data_test)

# r2 = test_result$score(msr("regr.rsq"))
result_rsq = test_result_smogn$score(msr("regr.rsq")) #R2=0.72
result_rmse = test_result_smogn$score(msr("regr.rmse"))#RMSE=731.11
result_mae = test_result_smogn$score(msr("regr.mae"))#MAE=347.30
result_mape = test_result_smogn$score(msr("regr.mape"))#MAPE=20.17
result_sse = test_result_smogn$score(msr("regr.sse"))#SSE=4.2E+07
mad(truth_response$`test_result_smogn$truth`, truth_response$`test_result_smogn$response`)#MAD=238.33


