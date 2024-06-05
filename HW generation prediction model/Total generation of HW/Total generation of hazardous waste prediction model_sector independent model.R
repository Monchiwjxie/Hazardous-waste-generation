# Importing R packages ----------------------------------------------------
library(mlr3verse)
library(tidyverse)
library(caret)
library(mltools)
library(data.table)
library(MLmetrics)

# OCM ---------------------------------------------------------------------
### Importing data ###
data_train = read.csv("HWsum_2614_smogn.csv") %>% 
  select(
    staff,
    process_5, process_6, process_7,
    process_8, process_9, process_10,process_11,
    pH, water,
    HWsum                                                                                                                                        
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_5 = as.factor(process_5),
    process_6 = as.factor(process_6),
    process_7 = as.factor(process_7),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))

data_test = read.csv("C2614_infer.csv") %>% 
  mutate(
    industry = as.factor(industry),
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_5 = as.factor(process_5),
    process_6 = as.factor(process_6),
    process_7 = as.factor(process_7),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))
### Model_training ###
task_sum = as_task_regr(data_train, target = "HWsum")

### Hyperparameter Optimization ###
# Ranger_par = lrn("regr.ranger")
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 11),
#   mtry = p_int(lower = 1, upper = 11))
# at = auto_tuner(
#   learner = Ranger_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rsq"),
#   search_space = search_space,
#   tuner = tnr("grid_search"),
#   term_evals = 12)
# set.seed(1234)
# at$train(task_sum, row_ids = 1:length(data_train$HWsum))
# at$tuning_result$learner_param_vals[[1]]
# $num.trees
# [1] 900
# $min.node.size
# [1] 3
# $mtry
# [1] 8
### Model training ###
set.seed(1234)
Ranger_model = lrn("regr.ranger",
                   num.trees = 900,
                   min.node.size = 3,
                   mtry = 8)
Ranger_model$train(task_sum, row_ids = 1:length(data_train$HWsum))
### Model testing ###
test_result = Ranger_model$predict_newdata(data_test)

test_truth = as.data.frame(test_result$truth)
test_response = as.data.frame(test_result$response)
truth_response = cbind(test_truth, test_response)
names(truth_response) <- c("truth", "response")

R2 <- R2(truth_response$truth, truth_response$response)
RMSE <- RMSE(truth_response$truth, truth_response$response)
MAD <- mad(truth_response$truth, truth_response$response)
MAE <- MAE(truth_response$truth, truth_response$response)
MAPE <- MAPE(truth_response$truth, truth_response$response)
MSE <- MSE(truth_response$truth, truth_response$response)
SSE <- sum((truth_response$truth - truth_response$response)^2)

Performance_parameters = data.table(
  R2, RMSE, MAD, MAE, MAPE, MSE, SSE
)
Performance_parameters

write.csv(truth_response,"C2614_tru_res.csv")

# CPM ---------------------------------------------------------------------
### Importing data ###
data_train = read.csv("HWsum_2631_smogn.csv") %>% 
  select(
    staff,
    process_5, process_6, process_7,
    process_8, process_9, process_10,process_11,
    N,
    HWsum                                                                                                                                        
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_5 = as.factor(process_5),
    process_6 = as.factor(process_6),
    process_7 = as.factor(process_7),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))

data_test = read.csv("C2631_infer.csv") %>% 
  mutate(
    industry = as.factor(industry),
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_5 = as.factor(process_5),
    process_6 = as.factor(process_6),
    process_7 = as.factor(process_7),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))
### Model_training ###
task_sum = as_task_regr(data_train, target = "HWsum")

### Hyperparameter Optimization ###
# Ranger_par = lrn("regr.ranger")
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 10),
#   mtry = p_int(lower = 1, upper = 10))
# at = auto_tuner(
#   learner = Ranger_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rsq"),
#   search_space = search_space,
#   tuner = tnr("grid_search"),
#   term_evals = 10)
# set.seed(1234)
# at$train(task_sum, row_ids = 1:length(data_train$HWsum))
# at$tuning_result$learner_param_vals[[1]]
# $num.trees
# [1] 900
# $min.node.size
# [1] 3
# $mtry
# [1] 5
### Model training ###
set.seed(1234)
Ranger_model = lrn("regr.ranger",
                   num.trees = 900,
                   min.node.size = 3,
                   mtry = 5)
Ranger_model$train(task_sum, row_ids = 1:length(data_train$HWsum))
### Model testing ###
test_result = Ranger_model$predict_newdata(data_test)

test_truth = as.data.frame(test_result$truth)
test_response = as.data.frame(test_result$response)
truth_response = cbind(test_truth, test_response)
names(truth_response) <- c("truth", "response")

R2 <- R2(truth_response$truth, truth_response$response)
RMSE <- RMSE(truth_response$truth, truth_response$response)
MAD <- mad(truth_response$truth, truth_response$response)
MAE <- MAE(truth_response$truth, truth_response$response)
MAPE <- MAPE(truth_response$truth, truth_response$response)
MSE <- MSE(truth_response$truth, truth_response$response)
SSE <- sum((truth_response$truth - truth_response$response)^2)

Performance_parameters = data.table(
  R2, RMSE, MAD, MAE, MAPE, MSE, SSE
)
Performance_parameters

write.csv(truth_response,"C2631_tru_res.csv")

# SCP ---------------------------------------------------------------------
### Importing data ###
data_train = read.csv("HWsum_2669_smogn.csv") %>% 
  select(
    staff,
    process_2, process_3, process_5, process_6, process_7,
    process_8, process_9, process_10,process_11,
    P, pH,
    HWsum                                                                                                                                        
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_5 = as.factor(process_2),
    process_6 = as.factor(process_3),
    process_5 = as.factor(process_5),
    process_6 = as.factor(process_6),
    process_7 = as.factor(process_7),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))


data_test = read.csv("C2669_infer.csv") %>% 
  mutate(
    industry = as.factor(industry),
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_5 = as.factor(process_5),
    process_6 = as.factor(process_6),
    process_7 = as.factor(process_7),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))
### Model_training ###
task_sum = as_task_regr(data_train, target = "HWsum")

### Hyperparameter Optimization ###
# Ranger_par = lrn("regr.ranger")
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 12),
#   mtry = p_int(lower = 1, upper = 12))
# at = auto_tuner(
#   learner = Ranger_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rsq"),
#   search_space = search_space,
#   tuner = tnr("grid_search"),
#   term_evals = 12)
# set.seed(1234)
# at$train(task_sum, row_ids = 1:length(data_train$HWsum))
# at$tuning_result$learner_param_vals[[1]]
# $num.trees
# [1] 680
# $min.node.size
# [1] 5
# $mtry
# [1] 8
### Model training ###
set.seed(1234)
Ranger_model = lrn("regr.ranger",
                   num.trees = 640,
                   min.node.size = 5,
                   mtry = 8)
Ranger_model$train(task_sum, row_ids = 1:length(data_train$HWsum))
### Model testing ###
test_result = Ranger_model$predict_newdata(data_test)

test_truth = as.data.frame(test_result$truth)
test_response = as.data.frame(test_result$response)
truth_response = cbind(test_truth, test_response)
names(truth_response) <- c("truth", "response")

R2 <- R2(truth_response$truth, truth_response$response)
RMSE <- RMSE(truth_response$truth, truth_response$response)
MAD <- mad(truth_response$truth, truth_response$response)
MAE <- MAE(truth_response$truth, truth_response$response)
MAPE <- MAPE(truth_response$truth, truth_response$response)
MSE <- MSE(truth_response$truth, truth_response$response)
SSE <- sum((truth_response$truth - truth_response$response)^2)

Performance_parameters = data.table(
  R2, RMSE, MAD, MAE, MAPE, MSE, SSE
)
Performance_parameters

write.csv(truth_response,"C2669_tru_res.csv")

# SRP ---------------------------------------------------------------------
### Importing data ###
data_train = read.csv("HWsum_3130_smogn.csv") %>% 
  select(
    staff,
    process_1, process_2, process_3,
    process_9, process_10,process_11,
    N, pH,
    Cr,
    HWsum                                                                                                                                        
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))

data_test = read.csv("C3130_infer.csv") %>% 
  mutate(
    industry = as.factor(industry),
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_5 = as.factor(process_5),
    process_6 = as.factor(process_6),
    process_7 = as.factor(process_7),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))
### Model_training ###
task_sum = as_task_regr(data_train, target = "HWsum")

### Hyperparameter Optimization ###
# Ranger_par = lrn("regr.ranger")
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 9),
#   mtry = p_int(lower = 1, upper = 9))
# at = auto_tuner(
#   learner = Ranger_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rsq"),
#   search_space = search_space,
#   tuner = tnr("grid_search"),
#   term_evals = 9)
# set.seed(1234)
# at$train(task_sum, row_ids = 1:length(data_train$HWsum))
# at$tuning_result$learner_param_vals[[1]]
# $num.trees
# [1] 460
# $min.node.size
# [1] 3
# $mtry
# [1] 6
### Model training ###
set.seed(1234)
Ranger_model = lrn("regr.ranger",
                   num.trees = 460,
                   min.node.size = 3,
                   mtry = 6)
Ranger_model$train(task_sum, row_ids = 1:length(data_train$HWsum))
### Model testing ###
test_result = Ranger_model$predict_newdata(data_test)

test_truth = as.data.frame(test_result$truth)
test_response = as.data.frame(test_result$response)
truth_response = cbind(test_truth, test_response)
names(truth_response) <- c("truth", "response")

R2 <- R2(truth_response$truth, truth_response$response)
RMSE <- RMSE(truth_response$truth, truth_response$response)
MAD <- mad(truth_response$truth, truth_response$response)
MAE <- MAE(truth_response$truth, truth_response$response)
MAPE <- MAPE(truth_response$truth, truth_response$response)
MSE <- MSE(truth_response$truth, truth_response$response)
SSE <- sum((truth_response$truth - truth_response$response)^2)

Performance_parameters = data.table(
  R2, RMSE, MAD, MAE, MAPE, MSE, SSE
)
Performance_parameters

write.csv(truth_response,"C3130_tru_res.csv")

# MWR ---------------------------------------------------------------------
### Importing data ###
data_train = read.csv("HWsum_3340_smogn.csv") %>% 
  select(
    staff,
    process_1, process_2, process_3, process_4,
    process_9, process_10,process_11,
    P, pH,
    Cu,
    HWsum                                                                                                                                        
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))

data_test = read.csv("C3340_infer.csv") %>% 
  mutate(
    industry = as.factor(industry),
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_5 = as.factor(process_5),
    process_6 = as.factor(process_6),
    process_7 = as.factor(process_7),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))
### Model_training ###
task_sum = as_task_regr(data_train, target = "HWsum")

### Hyperparameter Optimization ###
# Ranger_par = lrn("regr.ranger")
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 11),
#   mtry = p_int(lower = 1, upper = 11))
# at = auto_tuner(
#   learner = Ranger_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rsq"),
#   search_space = search_space,
#   tuner = tnr("grid_search"),
#   term_evals = 11)
# set.seed(1234)
# at$train(task_sum, row_ids = 1:length(data_train$HWsum))
# at$tuning_result$learner_param_vals[[1]]
# $num.trees
# [1] 500
# $min.node.size
# [1] 6
# $mtry
# [1] 5
### Model training ###
set.seed(1234)
Ranger_model = lrn("regr.ranger",
                   num.trees = 500,
                   min.node.size = 6,
                   mtry = 5)
Ranger_model$train(task_sum, row_ids = 1:length(data_train$HWsum))
### Model testing ###
test_result = Ranger_model$predict_newdata(data_test)

test_truth = as.data.frame(test_result$truth)
test_response = as.data.frame(test_result$response)
truth_response = cbind(test_truth, test_response)
names(truth_response) <- c("truth", "response")

R2 <- R2(truth_response$truth, truth_response$response)
RMSE <- RMSE(truth_response$truth, truth_response$response)
MAD <- mad(truth_response$truth, truth_response$response)
MAE <- MAE(truth_response$truth, truth_response$response)
MAPE <- MAPE(truth_response$truth, truth_response$response)
MSE <- MSE(truth_response$truth, truth_response$response)
SSE <- sum((truth_response$truth - truth_response$response)^2)

Performance_parameters = data.table(
  R2, RMSE, MAD, MAE, MAPE, MSE, SSE
)
Performance_parameters

write.csv(truth_response,"C3340_tru_res.csv")

# MST ---------------------------------------------------------------------
### Importing data ###
data_train = read.csv("HWsum_3360_smogn.csv") %>% 
  select(
    staff,
    process_1, process_2, process_3, process_4,
    process_9, process_10,process_11,
    NH3N, pH,
    Ni,
    HWsum                                                                                                                                        
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))

data_test = read.csv("C3360_infer.csv") %>% 
  mutate(
    industry = as.factor(industry),
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_5 = as.factor(process_5),
    process_6 = as.factor(process_6),
    process_7 = as.factor(process_7),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))
### Model_training ###
task_sum = as_task_regr(data_train, target = "HWsum")

### Hyperparameter Optimization ###
# Ranger_par = lrn("regr.ranger")
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 13),
#   mtry = p_int(lower = 1, upper = 13))
# at = auto_tuner(
#   learner = Ranger_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rsq"),
#   search_space = search_space,
#   tuner = tnr("grid_search"),
#   term_evals = 13)
# set.seed(1234)
# at$train(task_sum, row_ids = 1:length(data_train$HWsum))
# at$tuning_result$learner_param_vals[[1]]
# $num.trees
# [1] 900
# $min.node.size
# [1] 8
# $mtry
# [1] 8
### Model training ###
set.seed(1234)
Ranger_model = lrn("regr.ranger",
                   num.trees = 900,
                   min.node.size = 8,
                   mtry = 8)
Ranger_model$train(task_sum, row_ids = 1:length(data_train$HWsum))
### Model testing ###
test_result = Ranger_model$predict_newdata(data_test)

test_truth = as.data.frame(test_result$truth)
test_response = as.data.frame(test_result$response)
truth_response = cbind(test_truth, test_response)
names(truth_response) <- c("truth", "response")

R2 <- R2(truth_response$truth, truth_response$response)
RMSE <- RMSE(truth_response$truth, truth_response$response)
MAD <- mad(truth_response$truth, truth_response$response)
MAE <- MAE(truth_response$truth, truth_response$response)
MAPE <- MAPE(truth_response$truth, truth_response$response)
MSE <- MSE(truth_response$truth, truth_response$response)
SSE <- sum((truth_response$truth - truth_response$response)^2)

Performance_parameters = data.table(
  R2, RMSE, MAD, MAE, MAPE, MSE, SSE
)
Performance_parameters

write.csv(truth_response,"C3360_tru_res.csv")

# ECM ---------------------------------------------------------------------
### Importing data ###
data_train = read.csv("HWsum_3982_smogn.csv") %>% 
  select(
    staff,
    process_2, process_3, process_4, process_6, 
    process_8, process_9, process_10,process_11,
    pH, water,
    Ni,
    HWsum                                                                                                                                        
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_6 = as.factor(process_6),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))

data_test = read.csv("C3982_infer.csv") %>% 
  mutate(
    industry = as.factor(industry),
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_5 = as.factor(process_5),
    process_6 = as.factor(process_6),
    process_7 = as.factor(process_7),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))
### Model_training ###
task_sum = as_task_regr(data_train, target = "HWsum")

### Hyperparameter Optimization ###
# Ranger_par = lrn("regr.ranger")
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 13),
#   mtry = p_int(lower = 1, upper = 13))
# at = auto_tuner(
#   learner = Ranger_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rsq"),
#   search_space = search_space,
#   tuner = tnr("grid_search"),
#   term_evals = 13)
# set.seed(1234)
# at$train(task_sum, row_ids = 1:length(data_train$HWsum))
# at$tuning_result$learner_param_vals[[1]]
# $num.trees
# [1] 340
# $min.node.size
# [1] 1
# $mtry
# [1] 6
### Model training ###
set.seed(1234)
Ranger_model = lrn("regr.ranger",
                   num.trees = 340,
                   min.node.size = 1,
                   mtry = 6)
Ranger_model$train(task_sum, row_ids = 1:length(data_train$HWsum))
### Model testing ###
test_result = Ranger_model$predict_newdata(data_test)

test_truth = as.data.frame(test_result$truth)
test_response = as.data.frame(test_result$response)
truth_response = cbind(test_truth, test_response)
names(truth_response) <- c("truth", "response")

R2 <- R2(truth_response$truth, truth_response$response)
RMSE <- RMSE(truth_response$truth, truth_response$response)
MAD <- mad(truth_response$truth, truth_response$response)
MAE <- MAE(truth_response$truth, truth_response$response)
MAPE <- MAPE(truth_response$truth, truth_response$response)
MSE <- MSE(truth_response$truth, truth_response$response)
SSE <- sum((truth_response$truth - truth_response$response)^2)

Performance_parameters = data.table(
  R2, RMSE, MAD, MAE, MAPE, MSE, SSE
)
Performance_parameters

write.csv(truth_response,"C3982_tru_res.csv")

# BEG ---------------------------------------------------------------------
### Importing data ###
data_train = read.csv("HWsum_4417_smogn.csv") %>% 
  select(
    staff,
    process_2, process_3, process_6, 
    process_7, process_9, process_10,process_11,
    COD, pH,
    HWsum                                                                                                                                        
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_6 = as.factor(process_6),
    process_7 = as.factor(process_7),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))

data_test = read.csv("D4417_infer.csv") %>% 
  mutate(
    industry = as.factor(industry),
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_5 = as.factor(process_5),
    process_6 = as.factor(process_6),
    process_7 = as.factor(process_7),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))
### Model_training ###
task_sum = as_task_regr(data_train, target = "HWsum")

### Hyperparameter Optimization ###
# Ranger_par = lrn("regr.ranger")
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 11),
#   mtry = p_int(lower = 1, upper = 11))
# at = auto_tuner(
#   learner = Ranger_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rsq"),
#   search_space = search_space,
#   tuner = tnr("grid_search"),
#   term_evals = 11)
# set.seed(1234)
# at$train(task_sum, row_ids = 1:length(data_train$HWsum))
# at$tuning_result$learner_param_vals[[1]]
# $num.trees
# [1] 500
# $min.node.size
# [1] 2
# $mtry
# [1] 7
### Model training ###
set.seed(1234)
Ranger_model = lrn("regr.ranger",
                   num.trees = 500,
                   min.node.size = 2,
                   mtry = 7)
Ranger_model$train(task_sum, row_ids = 1:length(data_train$HWsum))
### Model testing ###
test_result = Ranger_model$predict_newdata(data_test)

test_truth = as.data.frame(test_result$truth)
test_response = as.data.frame(test_result$response)
truth_response = cbind(test_truth, test_response)
names(truth_response) <- c("truth", "response")

R2 <- R2(truth_response$truth, truth_response$response)
RMSE <- RMSE(truth_response$truth, truth_response$response)
MAD <- mad(truth_response$truth, truth_response$response)
MAE <- MAE(truth_response$truth, truth_response$response)
MAPE <- MAPE(truth_response$truth, truth_response$response)
MSE <- MSE(truth_response$truth, truth_response$response)
SSE <- sum((truth_response$truth - truth_response$response)^2)

Performance_parameters = data.table(
  R2, RMSE, MAD, MAE, MAPE, MSE, SSE
)
Performance_parameters

write.csv(truth_response,"D4417_tru_res.csv")

# EGU ---------------------------------------------------------------------
### Importing data ###
data_train = read.csv("HWsum_4419_smogn.csv") %>% 
  select(
    staff,
    process_2, process_3, process_4, process_6, 
    process_8, process_9, process_10,process_11,
    pH, P, NH3N,
    Cu,
    HWsum                                                                                                                                        
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_6 = as.factor(process_6),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))

data_test = read.csv("D4419_infer.csv") %>% 
  mutate(
    industry = as.factor(industry),
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_5 = as.factor(process_5),
    process_6 = as.factor(process_6),
    process_7 = as.factor(process_7),
    process_8 = as.factor(process_8),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))
### Model_training ###
task_sum = as_task_regr(data_train, target = "HWsum")

### Hyperparameter Optimization ###
# Ranger_par = lrn("regr.ranger")
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 12),
#   mtry = p_int(lower = 1, upper = 12))
# at = auto_tuner(
#   learner = Ranger_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rsq"),
#   search_space = search_space,
#   tuner = tnr("grid_search"),
#   term_evals = 12)
# set.seed(1234)
# at$train(task_sum, row_ids = 1:length(data_train$HWsum))
# at$tuning_result$learner_param_vals[[1]]
# $num.trees
# [1] 600
# $min.node.size
# [1] 4
# $mtry
# [1] 9
### Model training ###
set.seed(1234)
Ranger_model = lrn("regr.ranger",
                   num.trees = 600,
                   min.node.size = 4,
                   mtry = 9)
Ranger_model$train(task_sum, row_ids = 1:length(data_train$HWsum))
### Model testing ###
test_result = Ranger_model$predict_newdata(data_test)

test_truth = as.data.frame(test_result$truth)
test_response = as.data.frame(test_result$response)
truth_response = cbind(test_truth, test_response)
names(truth_response) <- c("truth", "response")

R2 <- R2(truth_response$truth, truth_response$response)
RMSE <- RMSE(truth_response$truth, truth_response$response)
MAD <- mad(truth_response$truth, truth_response$response)
MAE <- MAE(truth_response$truth, truth_response$response)
MAPE <- MAPE(truth_response$truth, truth_response$response)
MSE <- MSE(truth_response$truth, truth_response$response)
SSE <- sum((truth_response$truth - truth_response$response)^2)

Performance_parameters = data.table(
  R2, RMSE, MAD, MAE, MAPE, MSE, SSE
)
Performance_parameters

write.csv(truth_response,"D4419_tru_res.csv")
