# Importing R packages ----------------------------------------------------
library(mlr3verse)
library(tidyverse)
library(caret)
library(mltools)
library(data.table)
library(UBL)

# Model construction ------------------------------------------------------
#### Importing data ####
data <- read.csv("pathway../Model_HN_model_data_clean.csv") %>% 
  select(process_1,process_2,process_3,process_4,
         Size,
         Wastewater, COD, NH3N, TN, TP,  Cr, Cr6,
         HWsum) %>% 
  mutate(process_1 = as.factor(process_1),
         process_2 = as.factor(process_2),
         process_3 = as.factor(process_3),
         process_4 = as.factor(process_4),
         Size = as.factor(Size)
  )

set.seed(591)
sample = sample(length(data$HWsum), 0.2*length(data$HWsum))
data_test = data[sample,]
data_train = data[-sample,]

#### Data balance #### 
set.seed(1234)
data_train_smogn = SMOGNRegress(form = HWsum ~ .,
                                dat = data_train,
                                thr.rel = 0.24,
                                dist = "HEOM",
                                k = 5,
                                C.perc = "balance")

#### Task Creation ####
task_smogn = as_task_regr(data_train_smogn, target = "HWsum")

#### Hyperparameters ####
Ranger_model_smogn = lrn("regr.ranger",
                         num.trees = 500,
                         min.node.size = 4,
                         mtry = 6)
#### Model training ####
Ranger_model_smogn$train(task_smogn, row_ids = 1:length(data_train_smogn$HWsum))

#### Testing ####
test_result_smogn = Ranger_model_smogn$predict_newdata(data_test)
#### Truth & Response ####
test_truth = as.data.frame(test_result_smogn$truth)
test_response = as.data.frame(test_result_smogn$response)
truth_response = cbind(test_truth, test_response)

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


