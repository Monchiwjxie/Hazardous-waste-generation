library(mlr3verse)
library(tidyverse)
library(mlr3extralearners)
library(mltools)
library(caret)

# task creation -----------------------------------------------------------
task_sum = as_task_regr(data, target = "HWsum")
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
    interaction.depth = p_int(lower = 1, upper = 15),
    n.minobsinnode = p_int(lower = 1, upper = 10),
    shrinkage = p_dbl(lower = 0.05, upper = 0.5)
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
    min_child_weight = p_dbl(lower = 1, upper = 5),
    subsample = p_dbl(lower = 0.5, upper = 1),
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
    gamma = p_dbl(lower = 0.5, upper = 2),
	  epsilon = p_dbl(lower = 0.1, upper = 1),
    kernel = p_fct(c("polynomial", "radial")),
store_models = T
)

# Hyperparameter search ---------------------------------------------------
lmr$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
gbm$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
xgb$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
knn$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
ranger$train(task_sum, row_ids = 1:length(data_train_pre$HWsum))
svm $train(task_sum, row_ids = 1:length(data_train_pre$HWsum))



