library(UBL)
library(tidyverse)
library(data.table)
library(dplyr)
library(readr)
library(mlr3verse)
library(tidyverse)
library(caret)
library(mlr3extralearners)
library(mltools)

# Importing the training data set ------------------------------------------
data_class = data_train_pre %>% 
  select(-HWsum)

# MHW 0-1 Classification for train data set -------------------------------
for (i in 1:length(data_class$MHW)) {
  if (data_class$MHW[i] > 0) {
    data_class$MHWlogic[i] = 1
  } else{
    data_class$MHWlogic[i] = 0
  }
}
data_class = data_class %>% 
  select(-MHW)
data_class$MHWlogic = as.factor(data_class$MHWlogic)

# regr data balance -------------------------------------------------------
data_regr = data_train_pre %>% 
  select(-HWsum)
data_regr = data_regr[which(data_regr$MHW != 0),]
data_regr_smogn = SMOGNRegress(form = MHW ~ .,
                               dat = data_regr,
                               thr.rel = 0.24,
                               dist = "HEOM",
                               k = 5,
                               C.perc = "balance")


# MHW 0-1 Classification for test data set --------------------------------
for (i in 1:length(data_test$MHW)) {
  if (data_test$MHW[i] > 0) {
    data_test$MHWlogic[i] = 1
  } else{
    data_test$MHWlogic[i] = 0
  }
}
data_test$MHWlogic = as.factor(data_test$MHWlogic)

# Training a classification model -----------------------------------------
task_17_class = as_task_classif(data_class,
                                id = deparse(substitute(x)),
                                target = "MHWlogic")
Ranger_class = lrn("classif.ranger", 
                   num.trees = 960,
                   min.node.size = 14,
                   mtry = 7)
set.seed(123)
Ranger_class$predict_type = "prob"
Ranger_class$train(task_17_class, row_ids = 1:length(data_class$MHWlogic))


# Training a regression model ---------------------------------------------
task_17_regr = as_task_regr(data_regr_smogn,
                            target = "MHW")
Ranger_regr = lrn("regr.ranger", 
                  num.trees = 800,
                  min.node.size = 4,
                  mtry = 11)
Ranger_regr$train(task_17_regr, row_ids = 1:length(data_regr_smogn$MHW))

# test --------------------------------------------------------------------
class_result = Ranger_class$predict_newdata(data_test)
class_result$confusion

data_class_result = as.data.table(class_result$clone())
data_class_result = data_class_result[which(data_class_result$response==1),]

data_test_regr = data_test[data_class_result$row_ids,]
data_test_regr = data_test_regr [,-34]
regr_result = Ranger_regr$predict_newdata(data_test_regr)
regr_result$score(msr("regr.rsq"))
regr_result$score(msr("regr.rmse"))

regr_result_MHW = as.data.table(regr_result$clone())


# result ensemble ---------------------------------------------------------
row_id = as.vector(data_class_result$row_ids)
regr_response = as.vector(as.data.table(regr_result$clone())$response)
regr_table = data.frame(row_id, regr_response)
data_test$row_id = rep(1:length(data_test$MHW))
datt = merge(data_test,regr_table, by = "row_id", all = TRUE)
datt[is.na(datt)] = 0

MHW_all_data_smogn_test_result = datt %>% 
  select(MHW,regr_response)
R2(MHW_all_data_smogn_test_result$MHW,MHW_all_data_smogn_test_result$regr_response)
RMSE(MHW_all_data_smogn_test_result$MHW,MHW_all_data_smogn_test_result$regr_response)


# Predicting 4 industrial sector using ensemble models --------------------

## 3360 ##
for (i in 1:length(data_test_3360$MHW)) {
  if (data_test_3360$MHW[i] > 0) {
    data_test_3360$MHWlogic[i] = 1
  } else{
    data_test_3360$MHWlogic[i] = 0
  }
}
data_test_3360$MHWlogic = as.factor(data_test_3360$MHWlogic)
## 3130 ##
for (i in 1:length(data_test_3130$MHW)) {
  if (data_test_3130$MHW[i] > 0) {
    data_test_3130$MHWlogic[i] = 1
  } else{
    data_test_3130$MHWlogic[i] = 0
  }
}
data_test_3130$MHWlogic = as.factor(data_test_3130$MHWlogic)
## 3982 ##
for (i in 1:length(data_test_3982$MHW)) {
  if (data_test_3982$MHW[i] > 0) {
    data_test_3982$MHWlogic[i] = 1
  } else{
    data_test_3982$MHWlogic[i] = 0
  }
}
data_test_3982$MHWlogic = as.factor(data_test_3982$MHWlogic)
## 3340 ##
for (i in 1:length(data_test_3340$MHW)) {
  if (data_test_3340$MHW[i] > 0) {
    data_test_3340$MHWlogic[i] = 1
  } else{
    data_test_3340$MHWlogic[i] = 0
  }
}
data_test_3340$MHWlogic = as.factor(data_test_3340$MHWlogic)

### test 3360  ###
class_result_3360 = Ranger_class$predict_newdata(data_test_3360)
class_result_3360$confusion

data_class_result_3360 = as.data.table(class_result_3360$clone())
data_class_result_3360 = data_class_result_3360[which(data_class_result_3360$response==1),]

data_test_regr_3360 = data_test_3360[data_class_result_3360$row_ids,]
data_test_regr_3360 = data_test_regr_3360[,-34]
regr_result_3360 = Ranger_regr$predict_newdata(data_test_regr_3360)
regr_result_3360$score(msr("regr.rsq"))
regr_result_3360$score(msr("regr.rmse"))
# ensemble result #
row_id_3360 = as.vector(data_class_result_3360$row_ids)
regr_response_3360 = as.vector(as.data.table(regr_result_3360$clone())$response)
regr_table_3360 = data.frame(row_id_3360, regr_response_3360)
data_test_3360$row_id_3360 = rep(1:length(data_test_3360$MHW))
datt_3360 = merge(data_test_3360,regr_table_3360, by = "row_id_3360", all = TRUE)
datt_3360[is.na(datt_3360)] = 0

MHW_all_data_smogn_test_result_3360 = datt_3360 %>% 
  select(MHW,regr_response_3360)

R2(MHW_all_data_smogn_test_result_3360$MHW,MHW_all_data_smogn_test_result_3360$regr_response_3360)
RMSE(MHW_all_data_smogn_test_result_3360$MHW,MHW_all_data_smogn_test_result_3360$regr_response_3360)

### test 3130 ###
class_result_3130 = Ranger_class$predict_newdata(data_test_3130)
class_result_3130$confusion

data_class_result_3130 = as.data.table(class_result_3130$clone())
data_class_result_3130 = data_class_result_3130[which(data_class_result_3130$response==1),]

data_test_regr_3130 = data_test_3130[data_class_result_3130$row_ids,]
data_test_regr_3130 = data_test_regr_3130[,-34]
regr_result_3130 = Ranger_regr$predict_newdata(data_test_regr_3130)
regr_result_3130$score(msr("regr.rsq"))
regr_result_3130$score(msr("regr.rmse"))
# ensemble result #
row_id_3130 = as.vector(data_class_result_3130$row_ids)
regr_response_3130 = as.vector(as.data.table(regr_result_3130$clone())$response)
regr_table_3130 = data.frame(row_id_3130, regr_response_3130)
data_test_3130$row_id_3130 = rep(1:length(data_test_3130$MHW))
datt_3130 = merge(data_test_3130,regr_table_3130, by = "row_id_3130", all = TRUE)
datt_3130[is.na(datt_3130)] = 0

MHW_all_data_smogn_test_result_3130 = datt_3130 %>% 
  select(MHW,regr_response_3130)

R2(MHW_all_data_smogn_test_result_3130$MHW,MHW_all_data_smogn_test_result_3130$regr_response_3130)
RMSE(MHW_all_data_smogn_test_result_3130$MHW,MHW_all_data_smogn_test_result_3130$regr_response_3130)

### result 3982 ###
class_result_3982 = Ranger_class$predict_newdata(data_test_3982)
class_result_3982$confusion

data_class_result_3982 = as.data.table(class_result_3982$clone())
data_class_result_3982 = data_class_result_3982[which(data_class_result_3982$response==1),]

data_test_regr_3982 = data_test_3982[data_class_result_3982$row_ids,]
data_test_regr_3982 = data_test_regr_3982[,-34]
regr_result_3982 = Ranger_regr$predict_newdata(data_test_regr_3982)
regr_result_3982$score(msr("regr.rsq"))
regr_result_3982$score(msr("regr.rmse"))
# ensemble result #
row_id_3982 = as.vector(data_class_result_3982$row_ids)
regr_response_3982 = as.vector(as.data.table(regr_result_3982$clone())$response)
regr_table_3982 = data.frame(row_id_3982, regr_response_3982)
data_test_3982$row_id_3982 = rep(1:length(data_test_3982$MHW))
datt_3982 = merge(data_test_3982,regr_table_3982, by = "row_id_3982", all = TRUE)
datt_3982[is.na(datt_3982)] = 0

MHW_all_data_smogn_test_result_3982 = datt_3982 %>% 
  select(MHW,regr_response_3982)

R2(MHW_all_data_smogn_test_result_3982$MHW,MHW_all_data_smogn_test_result_3982$regr_response_3982)
RMSE(MHW_all_data_smogn_test_result_3982$MHW,MHW_all_data_smogn_test_result_3982$regr_response_3982)

### result 3340 ###
class_result_3340 = Ranger_class$predict_newdata(data_test_3340)
class_result_3340$confusion

data_class_result_3340 = as.data.table(class_result_3340$clone())
data_class_result_3340 = data_class_result_3340[which(data_class_result_3340$response==1),]

data_test_regr_3340 = data_test_3340[data_class_result_3340$row_ids,]
data_test_regr_3340 = data_test_regr_3340[,-34]
regr_result_3340 = Ranger_regr$predict_newdata(data_test_regr_3340)
regr_result_3340$score(msr("regr.rsq"))
regr_result_3340$score(msr("regr.rmse"))
# ensemble result #
row_id_3340 = as.vector(data_class_result_3340$row_ids)
regr_response_3340 = as.vector(as.data.table(regr_result_3340$clone())$response)
regr_table_3340 = data.frame(row_id_3340, regr_response_3340)
data_test_3340$row_id_3340 = rep(1:length(data_test_3340$MHW))
datt_3340 = merge(data_test_3340,regr_table_3340, by = "row_id_3340", all = TRUE)
datt_3340[is.na(datt_3340)] = 0

MHW_all_data_smogn_test_result_3340 = datt_3340 %>% 
  select(MHW,regr_response_3340)

R2(MHW_all_data_smogn_test_result_3340$MHW,MHW_all_data_smogn_test_result_3340$regr_response_3340)
RMSE(MHW_all_data_smogn_test_result_3340$MHW,MHW_all_data_smogn_test_result_3340$regr_response_3340)
