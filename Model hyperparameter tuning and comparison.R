library(mlr3verse)
library(tidyverse)
library(mlr3extralearners)
library(mltools)
library(caret)

# task creation -----------------------------------------------------------
task_sum = as_task_regr(data_train_pre, target = "HWsum")
# learner_all_data --------------------------------------------------------
lmr = auto_tuner(
  learner = lrn("regr.lm", predict_sets = c("train", "test")),
  resampling = rsmp("cv", folds = 10),
  store_models = T
)
  
gbm = auto_tuner(
  method = "grid_search",
  learner = lrn("regr.gbm", predict_sets = c("train", "test")),
  resampling = rsmp("cv", folds = 10),
  term_evals = 10,
  search_space = ps(
    n.trees = p_int(lower = 1, upper = 10,
                    trafo = function(x)100*x),
    interaction.depth = p_int(lower = 1, upper = 10),
    n.minobsinnode = p_int(lower = 1, upper = 10),
    shrinkage = p_dbl(lower = 0.001, upper = 5)
    ),
    store_models = T
)
  
xgb = auto_tuner(
  method = "grid_search",
  learner = lrn("regr.xgboost", predict_sets = c("train", "test")),
  resampling = rsmp("cv", folds = 10),
  term_evals = 10,
  search_space = ps(
    eta = p_dbl(lower = 0.1, upper = 1),
	max_depth = p_dbl(lower = 1, upper = 10),
    min_child_weight = p_dbl(lower = 0.01, upper = 5),
    subsample = p_dbl(lower = 0.5, upper = 1),
    colsample_bytree = p_dbl(lower = 0.7, upper = 1),
	gamma = p_dbl(lower = 0.1, upper = 1),
    nrounds = p_int(lower = 1, upper = 20, trafo = function(x)10*x)
  ),
  store_models = T
)
  
knn = auto_tuner(
  method = "grid_search",
  learner = lrn("regr.kknn", predict_sets = c("train", "test")),
  resampling = rsmp("cv", folds = 10),
  term_evals = 10,
  search_space = ps(
    k =  p_int(lower = 1, upper = 10, trafo = function(x)5*x),
    distance = p_dbl(lower = 1, upper = 6),
     kernel = p_fct(levels = c("rectangular", "gaussian",
                               "rank", "optimal"))
  ),
  store_models = T
)
  
ranger = auto_tuner(
  method = "grid_search",
  learner = lrn("regr.ranger", predict_sets = c("train", "test")),
  resampling = rsmp("cv", folds = 10),
   term_evals = 10,
  search_space = ps(
    num.trees = p_int(lower = 1, upper = 50,
                      trafo = function(x) 20 * x),
    min.node.size = p_int(lower = 1, upper = 15),
    mtry = p_int(lower = 1, upper = 15)
    ),
  store_models = T
)
  
svm = auto_tuner(
  method = "grid_search",
  learner = lrn("regr.svm", type = "eps-regression",
      predict_sets = c("train", "test")),
  resampling = rsmp("cv", folds = 10),
  term_evals = 10,
  search_space = ps(
    cost = p_dbl(lower = 1, upper = 10,
                      trafo = function(x) 5 * x),),
    gamma = p_dbl(lower = 0, upper = 2),
	epsilon = p_dbl(lower = 0.1, upper = 1),
    kernel = p_fct(c("polynomial", "radial"))
  ),
store_models = T
)

# Hyperparameter search ---------------------------------------------------
lmr$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
gbm$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
xgb$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
knn$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
ranger$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
svm $train(task_sum, row_ids = 1:length(data_train_pre$HWsum))

# Optimal hyperparameter setting ------------------------------------------
Lmr_model = lrn("regr.lm")
Gbm_model = lrn("regr.gbm")
Gbm_model$param$values = gbm$tuning_result$learner_param_vals[[1]]

Xgb_model = lrn("regr.xgboost")
Xgb_model$param$values = xgb$tuning_result$learner_param_vals[[1]]

Knn_model = lrn("regr.kknn")
Knn_model$param$values = knn$tuning_result$learner_param_vals[[1]]

Ranger_model = lrn("regr.ranger")
Ranger_model$param$values = Ranger$tuning_result$learner_param_vals[[1]]

Svm_model = lrn("regr.svm", type = "eps-regression")
Svm_model$param$values = svm$tuning_result$learner_param_vals[[1]]

# Model_training ----------------------------------------------------------
Lmr_model$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
Gbm_model$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
Xgb_model$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
Knn_model$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
Ranger_model$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
Svm_model$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))

# testing -----------------------------------------------------------------
Lmr_model_test = Lmr_model$predict_newdata(data_test)
Gbm_model_test = Gbm_model$predict_newdata(data_test)
Xgb_model_test = Xgb_model$predict_newdata(data_test)
Knn_model_test = Knn_model$predict_newdata(data_test)
Ranger_model_test = Ranger_model$predict_newdata(data_test)
Svm_model_test = Svm_model$predict_newdata(data_test)

# model evaluation --------------------------------------------------------
Lmr_model_test$score(msrs(c("regr.rsq", "regr.rmse")))
Gbm_model_test$score(msrs(c("regr.rsq", "regr.rmse")))
Xgb_model_test$score(msrs(c("regr.rsq", "regr.rmse")))
Knn_model_test$score(msrs(c("regr.rsq", "regr.rmse")))
Ranger_model_test$score(msrs(c("regr.rsq", "regr.rmse")))
Svm_model_test$score(msrs(c("regr.rsq", "regr.rmse")))

