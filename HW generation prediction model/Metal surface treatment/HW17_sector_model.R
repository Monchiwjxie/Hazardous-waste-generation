setwd("E:/HW_Review_2/绘图/Fig_S6_HW17_sector_model_performance")
# Importing R packages ----------------------------------------------------
library(mlr3verse)
library(tidyverse)
library(caret)
library(mltools)
library(data.table)
library(MLmetrics)


# SRP ---------------------------------------------------------------------
## training dataset ##
data_regr_train = read.csv("data_3130_regr_train_somgn.csv") %>% 
  select(
    staff,
    process_1, process_2, process_3,
    process_9, process_10,process_11,
    pH,NH3N,
    Ni,Cr,
    HW17 
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))

data_clas_train = read.csv("data_3130_clas_train.csv") %>% 
  select(
    staff,
    process_1, process_2, process_3,
    process_9, process_10,process_11,
    pH, NH3N,
    Ni,Cr,
    HW17logic
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11),
    HW17logic = as.factor(HW17logic))
## testing dataset ##
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

### Molel training ###
task_clas = as_task_classif(data_clas_train,
                            id = deparse(substitute(x)),
                            target = "HW17logic") #分类任务
task_regr = as_task_regr(data_regr_train,
                         target = "HW17") #回归任务

# ### Hyperparameter Optimization ###
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 12),
#   mtry = p_int(lower = 1, upper = 12))
# ### Classificaitong ###
# Ranger_class_par = lrn("classif.ranger")
# at_clas = auto_tuner(
#   learner = Ranger_class_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("classif.ce"),
#   search_space = search_space,
#   tuner = tnr("random_search"),
#   term_evals = 12)
# set.seed(1234)
# at_clas$train(task_clas, row_ids = 1:length(data_clas_train$HW17logic))
# ### Regression ###
# Ranger_regr_par = lrn("regr.ranger")
# at_regr = auto_tuner(
#   learner = Ranger_regr_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rmse"),
#   search_space = search_space,
#   tuner = tnr("random_search"),
#   term_evals = 12)
# set.seed(1234)
# at_regr$train(task_regr, row_ids = 1:length(data_regr_train$HW17))
# 
# at_clas$tuning_result$learner_param_vals[[1]]
# at_regr$tuning_result$learner_param_vals[[1]]

### Classification ###
task_clas = as_task_classif(data_clas_train,
                            id = deparse(substitute(x)),
                            target = "HW17logic")
set.seed(1234)
Ranger_clas = lrn("classif.ranger",
                  num.trees = 460,
                  min.node.size = 3,
                  mtry = 6)
Ranger_clas$predict_type = "prob"
Ranger_clas$train(task_clas, row_ids = 1:length(data_clas_train$HW17logic))
### Regression ###
task_regr = as_task_regr(data_regr_train,
                         target = "HW17")
set.seed(1234)
Ranger_regr = lrn("regr.ranger",
                  num.trees = 760,
                  min.node.size = 2,
                  mtry = 6)
Ranger_regr$train(task_regr, row_ids = 1:length(data_regr_train$HW17))

### Model test ###
## Classification ##
clas_result <- Ranger_clas$predict_newdata(data_test)
## Regression ##
regr_result <- Ranger_regr$predict_newdata(data_test)
data_test$response <- as.numeric(as.character(clas_result$response)) * regr_result$response
truth_response = data.frame(data_test$HW17, data_test$response)
names(truth_response) <- c("truth", "response")

R2 <- R2(data_test$HW17, data_test$response)
RMSE <- RMSE(data_test$HW17, data_test$response)
MAD <- mad(data_test$HW17, data_test$response)
MAE <- MAE(data_test$HW17, data_test$response)
MAPE <- MAPE(data_test$HW17, data_test$response)
MSE <- MSE(data_test$HW17, data_test$response)
SSE <- sum((data_test$HW17 - data_test$response)^2)

Performance_parameters = data.table(
  R2, RMSE, MAD, MAE, MAPE, MSE, SSE
)
Performance_parameters

write.csv(truth_response,"HW17_3130_tru_res.csv")

# MWR ---------------------------------------------------------------------
## training dataset ##
data_regr_train = read.csv("data_3340_regr_train_somgn.csv") %>% 
  select(
    staff,
    process_1, process_2, process_3, process_4, 
    process_9, process_10,
    pH, P,
    Ni,Zn,
    HW17 
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10))

data_clas_train = read.csv("data_3340_clas_train.csv") %>% 
  select(
    staff,
    process_1, process_2, process_3, process_4, 
    process_9, process_10,
    pH, P,
    Ni,Zn,
    HW17logic
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    HW17logic = as.factor(HW17logic))
## testing dataset ##
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

### Model training ###
task_clas = as_task_classif(data_clas_train,
                            id = deparse(substitute(x)),
                            target = "HW17logic") #分类任务
task_regr = as_task_regr(data_regr_train,
                         target = "HW17") #回归任务

# ### Hyperparameter Optimization ###
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 12),
#   mtry = p_int(lower = 1, upper = 12))
# ### Classificaitong ###
# Ranger_class_par = lrn("classif.ranger")
# at_clas = auto_tuner(
#   learner = Ranger_class_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("classif.ce"),
#   search_space = search_space,
#   tuner = tnr("random_search"),
#   term_evals = 12)
# set.seed(1234)
# at_clas$train(task_clas, row_ids = 1:length(data_clas_train$HW17logic))
# ### Regression ###
# Ranger_regr_par = lrn("regr.ranger")
# at_regr = auto_tuner(
#   learner = Ranger_regr_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rmse"),
#   search_space = search_space,
#   tuner = tnr("random_search"),
#   term_evals = 12)
# set.seed(1234)
# at_regr$train(task_regr, row_ids = 1:length(data_regr_train$HW17))
# 
# at_clas$tuning_result$learner_param_vals[[1]]
# at_regr$tuning_result$learner_param_vals[[1]]

### Classification ###
task_clas = as_task_classif(data_clas_train,
                            id = deparse(substitute(x)),
                            target = "HW17logic")
set.seed(1234)
Ranger_clas = lrn("classif.ranger",
                  num.trees = 760,
                  min.node.size = 2,
                  mtry = 6)
Ranger_clas$predict_type = "prob"
Ranger_clas$train(task_clas, row_ids = 1:length(data_clas_train$HW17logic))
### Regression ###
task_regr = as_task_regr(data_regr_train,
                         target = "HW17")
set.seed(1234)
Ranger_regr = lrn("regr.ranger",
                  num.trees = 720,
                  min.node.size = 6,
                  mtry = 6)
Ranger_regr$train(task_regr, row_ids = 1:length(data_regr_train$HW17))

### Model test ###
## Classification ##
clas_result <- Ranger_clas$predict_newdata(data_test)
## Regression ##
regr_result <- Ranger_regr$predict_newdata(data_test)
data_test$response <- as.numeric(as.character(clas_result$response)) * regr_result$response
truth_response = data.frame(data_test$HW17, data_test$response)
names(truth_response) <- c("truth", "response")

R2 <- R2(data_test$HW17, data_test$response)
RMSE <- RMSE(data_test$HW17, data_test$response)
MAD <- mad(data_test$HW17, data_test$response)
MAE <- MAE(data_test$HW17, data_test$response)
MAPE <- MAPE(data_test$HW17, data_test$response)
MSE <- MSE(data_test$HW17, data_test$response)
SSE <- sum((data_test$HW17 - data_test$response)^2)

Performance_parameters = data.table(
  R2, RMSE, MAD, MAE, MAPE, MSE, SSE
)
Performance_parameters

write.csv(truth_response,"HW17_3340_tru_res.csv")


# MST ---------------------------------------------------------------------
## training dataset ##
data_regr_train = read.csv("data_3360_regr_train_somgn.csv") %>% 
  select(
    staff,
    process_1, process_2, process_3, 
    process_9, process_10, process_11, 
    pH, water, 
    Ni,Cu, Fe, Cr, Cr6,
    HW17 
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))

data_clas_train = read.csv("data_3360_clas_train.csv") %>% 
  select(
    staff,
    process_1, process_2, process_3, 
    process_9, process_10, process_11, 
    pH, water, 
    Ni,Cu, Fe, Cr, Cr6,
    HW17logic
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))
## testing dataset ##
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

### Model training ###
task_clas = as_task_classif(data_clas_train,
                            id = deparse(substitute(x)),
                            target = "HW17logic") #分类任务
task_regr = as_task_regr(data_regr_train,
                         target = "HW17") #回归任务

### Hyperparameter Optimization ###
search_space = ps(
  num.trees = p_int(lower = 1, upper = 50,
                    trafo = function(x) 20 * x),
  min.node.size = p_int(lower = 1, upper = 14),
  mtry = p_int(lower = 1, upper = 14))
### Classificaitong ###
Ranger_class_par = lrn("classif.ranger")
at_clas = auto_tuner(
  learner = Ranger_class_par,
  resampling = rsmp("cv", folds = 10),
  measure = msr("classif.ce"),
  search_space = search_space,
  tuner = tnr("random_search"),
  term_evals = 14)
set.seed(1234)
at_clas$train(task_clas, row_ids = 1:length(data_clas_train$HW17logic))
### Regression ###
Ranger_regr_par = lrn("regr.ranger")
at_regr = auto_tuner(
  learner = Ranger_regr_par,
  resampling = rsmp("cv", folds = 10),
  measure = msr("regr.rmse"),
  search_space = search_space,
  tuner = tnr("random_search"),
  term_evals = 14)
set.seed(1234)
at_regr$train(task_regr, row_ids = 1:length(data_regr_train$HW17))

at_clas$tuning_result$learner_param_vals[[1]]
at_regr$tuning_result$learner_param_vals[[1]]

### Classification ###
task_clas = as_task_classif(data_clas_train,
                            id = deparse(substitute(x)),
                            target = "HW17logic")
set.seed(1234)
Ranger_clas = lrn("classif.ranger",
                  num.trees = 660,
                  min.node.size = 6,
                  mtry = 4)
Ranger_clas$predict_type = "prob"
Ranger_clas$train(task_clas, row_ids = 1:length(data_clas_train$HW17logic))
### Regression ###
task_regr = as_task_regr(data_regr_train,
                         target = "HW17")
set.seed(1234)
Ranger_regr = lrn("regr.ranger",
                  num.trees = 620,
                  min.node.size = 8,
                  mtry = 8)
Ranger_regr$train(task_regr, row_ids = 1:length(data_regr_train$HW17))

### Model test ###
## Classification ##
clas_result <- Ranger_clas$predict_newdata(data_test)
## Regression ##
regr_result <- Ranger_regr$predict_newdata(data_test)
data_test$response <- as.numeric(as.character(clas_result$response)) * regr_result$response
truth_response = data.frame(data_test$HW17, data_test$response)
names(truth_response) <- c("truth", "response")

R2 <- R2(data_test$HW17, data_test$response)
RMSE <- RMSE(data_test$HW17, data_test$response)
MAD <- mad(data_test$HW17, data_test$response)
MAE <- MAE(data_test$HW17, data_test$response)
MAPE <- MAPE(data_test$HW17, data_test$response)
MSE <- MSE(data_test$HW17, data_test$response)
SSE <- sum((data_test$HW17 - data_test$response)^2)

Performance_parameters = data.table(
  R2, RMSE, MAD, MAE, MAPE, MSE, SSE
)
Performance_parameters

write.csv(truth_response,"HW17_3360_tru_res.csv")

# ECM ---------------------------------------------------------------------
## training dataset ##
data_regr_train = read.csv("data_3982_regr_train_somgn.csv") %>% 
  select(
    staff,
    process_1, process_2, process_3, process_4, process_6,
    process_9, process_10, process_11, 
    pH, COD, NH3N, 
    Ni,Cu,
    HW17 
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_6 = as.factor(process_6),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))

data_clas_train = read.csv("data_3982_clas_train.csv") %>% 
  select(
    staff,
    process_1, process_2, process_3, process_4, process_6,
    process_9, process_10, process_11, 
    pH, COD, NH3N, 
    Ni,Cu,
    HW17logic
  ) %>% 
  mutate(
    staff = as.factor(staff),
    process_1 = as.factor(process_1),
    process_2 = as.factor(process_2),
    process_3 = as.factor(process_3),
    process_4 = as.factor(process_4),
    process_6 = as.factor(process_6),
    process_9 = as.factor(process_9),
    process_10 = as.factor(process_10),
    process_11 = as.factor(process_11))
## testing dataset ##
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

### Model training ###
task_clas = as_task_classif(data_clas_train,
                            id = deparse(substitute(x)),
                            target = "HW17logic") #分类任务
task_regr = as_task_regr(data_regr_train,
                         target = "HW17") #回归任务

### Hyperparameter Optimization ###
search_space = ps(
  num.trees = p_int(lower = 1, upper = 50,
                    trafo = function(x) 20 * x),
  min.node.size = p_int(lower = 1, upper = 14),
  mtry = p_int(lower = 1, upper = 14))
### Classificaitong ###
Ranger_class_par = lrn("classif.ranger")
at_clas = auto_tuner(
  learner = Ranger_class_par,
  resampling = rsmp("cv", folds = 10),
  measure = msr("classif.ce"),
  search_space = search_space,
  tuner = tnr("random_search"),
  term_evals = 14)
set.seed(1234)
at_clas$train(task_clas, row_ids = 1:length(data_clas_train$HW17logic))
### Regression ###
Ranger_regr_par = lrn("regr.ranger")
at_regr = auto_tuner(
  learner = Ranger_regr_par,
  resampling = rsmp("cv", folds = 10),
  measure = msr("regr.rmse"),
  search_space = search_space,
  tuner = tnr("random_search"),
  term_evals = 14)
set.seed(1234)
at_regr$train(task_regr, row_ids = 1:length(data_regr_train$HW17))

at_clas$tuning_result$learner_param_vals[[1]]
at_regr$tuning_result$learner_param_vals[[1]]

### Classification ###
task_clas = as_task_classif(data_clas_train,
                            id = deparse(substitute(x)),
                            target = "HW17logic")
set.seed(1234)
Ranger_clas = lrn("classif.ranger",
                  num.trees = 620,
                  min.node.size = 8,
                  mtry = 6)
Ranger_clas$predict_type = "prob"
Ranger_clas$train(task_clas, row_ids = 1:length(data_clas_train$HW17logic))
### Regression ###
task_regr = as_task_regr(data_regr_train,
                         target = "HW17")
set.seed(1234)
Ranger_regr = lrn("regr.ranger",
                  num.trees = 500,
                  min.node.size = 8,
                  mtry = 4)
Ranger_regr$train(task_regr, row_ids = 1:length(data_regr_train$HW17))

### Model test ###
## Classification ##
clas_result <- Ranger_clas$predict_newdata(data_test)
## Regression ##
regr_result <- Ranger_regr$predict_newdata(data_test)
data_test$response <- as.numeric(as.character(clas_result$response)) * regr_result$response
truth_response = data.frame(data_test$HW17, data_test$response)
names(truth_response) <- c("truth", "response")

R2 <- R2(data_test$HW17, data_test$response)
RMSE <- RMSE(data_test$HW17, data_test$response)
MAD <- mad(data_test$HW17, data_test$response)
MAE <- MAE(data_test$HW17, data_test$response)
MAPE <- MAPE(data_test$HW17, data_test$response)
MSE <- MSE(data_test$HW17, data_test$response)
SSE <- sum((data_test$HW17 - data_test$response)^2)

Performance_parameters = data.table(
  R2, RMSE, MAD, MAE, MAPE, MSE, SSE
)
Performance_parameters

write.csv(truth_response,"HW17_3982_tru_res.csv")
