library(mlr3verse)
library(tidyverse)
library(caret)
library(mlr3extralearners)
library(mltools)


# sector_data -------------------------------------------------------------
scetor_data_train = data_train_3360 %>% 
  select(-MHW)#Can be replaced with other industrial sectors
scetor_data_test = data_test_3360
# data_balance ------------------------------------------------------------
data_train = SMOGNRegress(form = HWsum ~ .,
                          dat = scetor_data_train,
                          thr.rel = 0.24, # Changes based on industrial data characteristics
                          dist = "HEOM",
                          k = 5,
                          C.perc = "balance")

# sector-independent model ------------------------------------------------
#### RF model to predict the total generation quantity of HW ####
task_sum = as_task_regr(data_train, target = "HWsum")
Ranger_model = lrn("regr.ranger",
                   num.trees =700,#Optimal hyperparameter search based on ten-fold cross-validation
                   min.node.size = 6,
                   mtry = 22)
Ranger_model$train(task_sum, row_ids = 1:length(data_train$HWsum))

# test --------------------------------------------------------------------
test_result = Ranger_model$predict_newdata(scetor_data_test)
test_result$score(msr("regr.rsq"))
test_result$score(msr("regr.rmse"))

#### Ensemble model to predict the  generation quantity of MHW ####
data_class = scetor_data_train %>% 
  select(-HWsum)

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

for (i in 1:length(scetor_data_test$MHW)) {
  if (scetor_data_test$MHW[i] > 0) {
    scetor_data_test$MHWlogic[i] = 1
  } else{
    scetor_data_test$MHWlogic[i] = 0
  }
}
scetor_data_test$MHWlogic = as.factor(scetor_data_test$MHWlogic)

data_regr_prosmogn = scetor_data_train[which(scetor_data_train$MHW != 0),]
#### data balance ####
set.seed(1234)
data_regr = SMOGNRegress(form = MHW ~ .,
                         dat = data_regr_prosmogn,
                         thr.rel = 0.24,
                         dist = "HEOM",
                         k = 5,
                         C.perc = "balance")

#### classification model ####
task_class = as_task_classif(data_class,
                              id = deparse(substitute(x)),
                              target = "MHWlogic")
Ranger_class = lrn("classif.ranger", 
                   num.trees = 980,
                   min.node.size = 14,
                   mtry = 2,
                   predict_type = "prob")
Ranger_class$train(task_class, row_ids = 1:length(data_class$MHWlogic))

#### Regression model #### 
task_regr = as_task_regr(data_regr,
                         target = "MHW")
Ranger_regr = lrn("regr.ranger", 
                  num.trees = 960,
                  min.node.size = 3,
                  mtry = 11)
Ranger_regr$train(task_regr, row_ids = 1:length(data_regr$MHW))
# test  -----------------------------------------------------------------
class_result = Ranger_class$predict_newdata(scetor_data_test)
class_result$confusion

data_class_result = as.data.table(class_result$clone())
data_class_result = data_class_result[which(data_class_result$response==1),]

data_test_regr = scetor_data_test[data_class_result$row_ids,]
regr_result = Ranger_regr$predict_newdata(data_test_regr)
regr_result$score(msr("regr.rsq"))
regr_result$score(msr("regr.rmse"))

#ensemble result
row_id = as.vector(data_class_result$row_ids)
regr_response = as.vector(as.data.table(regr_result$clone())$response)
regr_table = data.frame(row_id, regr_response)
scetor_data_test$row_id = rep(1:length(scetor_data_test$MHW))
datt = merge(scetor_data_test,regr_table, by = "row_id", all = TRUE)
datt[is.na(datt)] = 0

MHW_smogn_test_result = datt %>% 
  select(MHW,regr_response)

R2(MHW_smogn_test_result$MHW,MHW_smogn_test_result$regr_response)
RMSE(MHW_smogn_test_result$MHW,MHW_smogn_test_result$regr_response)