# R包导入 --------------------------------------------------------------------
library(mlr3verse)
library(tidyverse)
library(caret)
library(mlr3extralearners)
library(mltools)
library(data.table)
# 模型构建 --------------------------------------------------------------------
#### 数据导入—训练数据及测试用例 ####
HWsum_4419_SMOGN = read.csv("HWsum_4419_smogn.csv")
data_test = read.csv("HWsum_4419_test.csv")

#### 选择变量 ####
data_model_train = HWsum_4419_SMOGN %>% 
  select(staff,
         process_1,process_2,process_3,process_4,process_5,
         process_6,process_7,process_8,process_9,process_10,
         process_11,
         COD,pH,water,P,NH3N,N,
         Cr6,Cr,Ni,Fe,Cu,Zn,
         HWsum)

#### 创建任务 ####
task_sum = as_task_regr(data_model_train, target = "HWsum")

# ####超参数搜索 ####
# Ranger_par = lrn("regr.ranger")
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 31),
#   mtry = p_int(lower = 1, upper = 31))
# at = auto_tuner(
#   learner = Ranger_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rmse"),
#   search_space = search_space,
#   tuner = tnr("random_search"),
#   term_evals = 31)
# set.seed(615)
# at$train(task_sum, row_ids = 1:length(data_model_train$HWsum))

#### 选择最优参数建模 ####
Ranger_model = lrn("regr.ranger",
                   num.trees = 600,
                   min.node.size = 4,
                   mtry = 9)
# Ranger_model$param_set$values = at$tuning_result$learner_param_vals[[1]]
Ranger_model$train(task_sum, row_ids = 1:length(data_model_train$HWsum))
#### 测试集计算 ####
test_result = Ranger_model$predict_newdata(data_test)

#### 测试集真实值-预测值结果导出 ####
test_truth = as.data.frame(test_result$truth)
test_response = as.data.frame(test_result$response)
truth_response = cbind(test_truth, test_response)
write.csv(truth_response, "HWsum_4419_tru_res.csv")

#### 模型性能与超参数结果导出 ####
result_rsq = test_result$score(msr("regr.rsq"))
result_rmse = test_result$score(msr("regr.rmse"))
result_mae = test_result$score(msr("regr.mae"))
result_mape = test_result$score(msr("regr.mape"))
result_sse = test_result$score(msr("regr.sse"))
model_param = as.data.table(Ranger_model$param_set$values)
Performance_parameters = data.table(result_rsq, result_rmse, result_mae,result_mape,result_sse,
                                    model_param)
write.csv(Performance_parameters, "HWsum_4419_Performance_parameters.csv")





