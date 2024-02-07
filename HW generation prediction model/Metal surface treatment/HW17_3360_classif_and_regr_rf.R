# R包导入 --------------------------------------------------------------------
library(mlr3verse)
library(tidyverse)
library(caret)
library(mlr3extralearners)
library(mltools)
library(data.table)

# 数据导入 --------------------------------------------------------------------
#### 分类训练集及变量筛选 ####
data_clas_train = read.csv("one_typical_3360_classification_train.csv") %>% 
  select(staff,
         process_1,process_2,process_3,process_4,process_5,
         process_6,process_7,process_8,process_9,process_10,
         process_11,
         COD,pH,water,P,NH3N,N,
         Cr6,Cr,Ni,Fe,Cu,Zn,
         HW17logic)
#### 回归训练集及变量筛选 ####
data_regr_train = read.csv("one_typical_3360_Regression_train.csv")%>% 
  select(staff,
         process_1,process_2,process_3,process_4,process_5,
         process_6,process_7,process_8,process_9,process_10,
         process_11,
         COD,pH,water,P,NH3N,N,
         Cr6,Cr,Ni,Fe,Cu,Zn,
         HW17)
#### 测试集导入 ####
data_test = read.csv("HW17_3360_test.csv")
for (i in 1:length(data_test$HW17)) {
  if (data_test$HW17[i] > 0) {
    data_test$HW17logic[i] = 1
  } else{
    data_test$HW17logic[i] = 0
  }
}
data_test$HW17logic = as.factor(data_test$HW17logic)

# 模型构建 --------------------------------------------------------------------
# #### 任务构建 ####
# task_clas = as_task_classif(data_clas_train,
#                             id = deparse(substitute(x)),
#                             target = "HW17logic") #分类任务
# task_regr = as_task_regr(data_regr_train,
#                          target = "HW17") #回归任务
# 
# #### 参数搜索空间设置 ####
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 31),
#   mtry = p_int(lower = 1, upper = 31))
# #### 分类模型调参 ####
# Ranger_class_par = lrn("classif.ranger")
# at_clas = auto_tuner(
#   learner = Ranger_class_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("classif.ce"),
#   search_space = search_space,
#   tuner = tnr("random_search"),
#   term_evals = 31)
# set.seed(615)
# at_clas$train(task_clas, row_ids = 1:length(data_clas_train$HW17logic))
# #### 回归模型调参 ####
# Ranger_regr_par = lrn("regr.ranger")
# at_regr = auto_tuner(
#   learner = Ranger_regr_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rmse"),
#   search_space = search_space,
#   tuner = tnr("random_search"),
#   term_evals = 32)
# set.seed(615)
# at_regr$train(task_regr, row_ids = 1:length(data_regr_train$HW17))

#### 分类模型构建 #### 
Ranger_clas = lrn("classif.ranger",
                  num.trees = 460,
                  min.node.size = 6,
                  mtry = 15)
# Ranger_clas$param_set$values = at_clas$tuning_result$learner_param_vals[[1]]
Ranger_clas$train(task_clas, row_ids = 1:length(data_clas_train$HW17logic))
#### 回归模型构建 #### 
Ranger_regr = lrn("regr.ranger",
                  num.trees = 620,
                  min.node.size = 2,
                  mtry = 15)
# Ranger_regr$param_set$values = at_regr$tuning_result$learner_param_vals[[1]]
Ranger_regr$train(task_regr, row_ids = 1:length(data_regr_train$HW17))

# 模型验证 --------------------------------------------------------------------
#### 分类模型混淆矩阵输出 ####
clas_result = Ranger_clas$predict_newdata(data_test)
confusion_matrix = data.table(clas_result$confusion)
clas_acc = clas_result$score(msr("classif.acc"))
clas_auc = clas_result$score(msr("classif.auc"))
clas_precision = clas_result$score(msr("classif.precision"))
clas_recall = clas_result$score(msr("classif.recall"))
clas_f1 = clas_result$score(msr("classif.fbeta"))
write.csv(confusion_matrix, "one_typical_3360_confusion_matrix.csv")

#### 计算测试集中非0数据的回归结果 ####
data_clas_result = as.data.table(clas_result$clone())
data_clas_result = data_clas_result[which(data_clas_result$response==1),]
data_test_regr = data_test[data_clas_result$row_ids,]
data_test_regr = data_test_regr[,-34]
regr_result = Ranger_regr$predict_newdata(data_test_regr)
regr_rsq = regr_result$score(msr("regr.rsq"))
regr_rmse = regr_result$score(msr("regr.rmse"))
regr_mae = regr_result$score(msr("regr.mae"))
regr_mape = regr_result$score(msr("regr.mape"))
regr_sse = regr_result$score(msr("regr.sse"))
regr_truth_response = as.data.table(regr_result$clone())
write.csv(regr_truth_response, "one_typical_3360_truth_response.csv")

#### 模型超参数、分类结果和回归结果导出 ####
clas_param = as.data.table(Ranger_clas$param_set$values)
regr_param = as.data.table(Ranger_regr$param_set$values)
Performance_parameters = data.table(regr_rsq, regr_rmse, regr_mae, regr_mape, regr_sse,
                                    clas_acc, clas_auc, clas_precision, clas_recall, clas_f1,
                                    clas_param, regr_param)
write.csv(Performance_parameters, "one_typical_3360_Performance_parameters.csv")

#### 分类回归结果合并后导出 ####
row_id = as.vector(data_clas_result$row_ids)
regr_response = as.vector(as.data.table(regr_result$clone())$response)
regr_table = data.frame(row_id, regr_response)
data_test$row_id = rep(1:length(data_test$HW17))
datt = merge(data_test,regr_table, by = "row_id", all = TRUE)
datt[is.na(datt)] = 0
HW17_all_data_smogn_test_result = datt %>% 
  select(HW17,regr_response)
write.csv(HW17_all_data_smogn_test_result, "one_typical_3360_alldata_truth_response.csv")


