library(mlr3verse)
library(tidyverse)
library(mlr3extralearners)
library(mltools)
library(caret)


# Model trian -------------------------------------------------------------
## before balance ##
task_sum_before_balance = as_task_regr(data_train_pre, target = "HWsum")
Ranger_model_before_balance = lrn("regr.ranger",
                                   num.trees =760,
                                   min.node.size = 6,
                                   mtry = 8)
Ranger_model_before_balance$train(task_sum_before_balance, 
                                  row_ids = 1:length(data_train_pre$HWsum))

## before balance ##
task_sum_after_balance = as_task_regr(data_train_smogn, target = "HWsum")
Ranger_model_after_balance = lrn("regr.ranger",
                                  num.trees =800,
                                  min.node.size = 6,
                                  mtry = 11)
Ranger_model_after_balance$train(task_sum_after_balance, 
                                 row_ids = 1:length(data_train_pre$HWsum))
# testing -----------------------------------------------------------------
Ranger_model_before_balance_test = Ranger_model_before_balance$predict_newdata(data_test)
Ranger_model_after_balance_test = Ranger_model_after_balance$predict_newdata(data_test)
# model evaluation --------------------------------------------------------
Ranger_model_before_balance_test$score(msrs(c("regr.rsq", "regr.rmse")))
Ranger_model_after_balance_test$score(msrs(c("regr.rsq", "regr.rmse")))

# Predict sector test data sets -------------------------------------------
scetor_data = data_test_3360 #Can be replaced with other industrial sectors
Sector_evaluation = Ranger_model_after_balance$predict_newdata(scetor_data)
Sector_evaluation$score(msrs(c("regr.rsq", "regr.rmse")))

