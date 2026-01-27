# Importing R packages ----------------------------------------------------
library(mlr3verse)
library(tidyverse)
library(caret)
library(mlr3extralearners)
library(mltools)
library(data.table)

# Model construction ------------------------------------------------------
#### Importing data ####
HWsum_combined_SMOGN = read.csv("pathway../HWsum_combined_smogn.csv")
data_test = read.csv("pathway../HWsum_combined_test.csv")

#### Select features ####
data_model_train = HWsum_combined_SMOGN %>% 
  select(industry,staff,
         process_1,process_2,process_3,process_4,process_5,
         process_6,process_7,process_8,process_9,process_10,
         process_11,
         COD,pH,water,P,NH3N,N,
         Cr6,Cr,Ni,Fe,Cu,Zn,
         HWsum)

#### Task creation ####
task_sum = as_task_regr(data_model_train, target = "HWsum")

# #### Hyperparameter ####
# Ranger_par = lrn("regr.ranger")
# search_space = ps(
#   num.trees = p_int(lower = 1, upper = 50,
#                     trafo = function(x) 20 * x),
#   min.node.size = p_int(lower = 1, upper = 32),
#   mtry = p_int(lower = 1, upper = 32))
# at = auto_tuner(
#   learner = Ranger_par,
#   resampling = rsmp("cv", folds = 10),
#   measure = msr("regr.rmse"),
#   search_space = search_space,
#   tuner = tnr("random_search"),
#   term_evals = 32)
# set.seed(615)
# at$train(task_sum, row_ids = 1:length(data_model_train$HWsum))

#### Modeling with optimal hyperparameters ####
Ranger_model = lrn("regr.ranger",
                   num.trees = 900,
                   min.node.size = 8,
                   mtry = 8)
# Ranger_model$param_set$values = at$tuning_result$learner_param_vals[[1]]
Ranger_model$train(task_sum, row_ids = 1:length(data_model_train$HWsum))

#### Testing ####
test_result = Ranger_model$predict_newdata(data_test)

#### Truth & Response ####
test_truth = as.data.frame(test_result_smogn$truth)
test_response = as.data.frame(test_result_smogn$response)
truth_response = cbind(test_truth, test_response)
write.csv(truth_response, "HWsum_combined_tru_res.csv")

#### Model performance metrics ####
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