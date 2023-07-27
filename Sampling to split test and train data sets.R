library(tidyverse)

# Data import -------------------------------------------------------------
data = read.csv("model_data.csv", encoding = "UTF-8", header = T) %>% 
  select(-c(Staff))
# select total quantity of HW ---------------------------------------------
data_sum = data %>% 
  select(-c(MHW)) %>% 
  mutate(Industrial_sector = as.factor(Industrial_sector),
         Firm_scale = as.factor(Firm_scale),
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
         process_11 = as.factor(process_11),
         process_12 = as.factor(process_12),
         process_13 = as.factor(process_13),
         process_14 = as.factor(process_14),
         process_15 = as.factor(process_15),
         process_16 = as.factor(process_16),
         process_17 = as.factor(process_17),
         process_18 = as.factor(process_18))

# split data to 8_2 ----------------------------------------------------------
set.seed(1234)
#### 3360 ####
data_sum_3360 = data_sum %>% 
  filter(Industrial_sector == "3360") 
#sampling
sample_3360 = sample(length(data_sum_3360$HWsum), 0.2*length(data_sum_3360$HWsum))
data_test_3360 = data_sum_3360[sample_3360,]
data_train_3360 = data_sum_3360[-sample_3360,]
#### 2614 ####
data_sum_2614 = data_sum %>% 
  filter(Industrial_sector == "2614") 
#sampling
sample_2614 = sample(length(data_sum_2614$HWsum), 0.2*length(data_sum_2614$HWsum))
data_test_2614 = data_sum_2614[sample_2614,]
data_train_2614 = data_sum_2614[-sample_2614,]
#### 2631 ####
data_sum_2631 = data_sum %>% 
  filter(Industrial_sector == "2631") 
#sampling
sample_2631 = sample(length(data_sum_2631$HWsum), 0.2*length(data_sum_2631$HWsum))
data_test_2631 = data_sum_2631[sample_2631,]
data_train_2631 = data_sum_2631[-sample_2631,]
#### 3130 ####
data_sum_3130 = data_sum %>% 
  filter(Industrial_sector == "3130") 
#sampling
sample_3130 = sample(length(data_sum_3130$HWsum), 0.2*length(data_sum_3130$HWsum))
data_test_3130 = data_sum_3130[sample_3130,]
data_train_3130 = data_sum_3130[-sample_3130,]
#### 3982 ####
data_sum_3982 = data_sum %>% 
  filter(Industrial_sector == "3982") 
#sampling
sample_3982 = sample(length(data_sum_3982$HWsum), 0.2*length(data_sum_3982$HWsum))
data_test_3982 = data_sum_3982[sample_3982,]
data_train_3982 = data_sum_3982[-sample_3982,]
#### 2669 ####
data_sum_2669 = data_sum %>% 
  filter(Industrial_sector == "2669") 
#sampling
sample_2669 = sample(length(data_sum_2669$HWsum), 0.2*length(data_sum_2669$HWsum))
data_test_2669 = data_sum_2669[sample_2669,]
data_train_2669 = data_sum_2669[-sample_2669,]
#### 3340 ####
data_sum_3340 = data_sum %>% 
  filter(Industrial_sector == "3340") 
#sampling
sample_3340 = sample(length(data_sum_3340$HWsum), 0.2*length(data_sum_3340$HWsum))
data_test_3340 = data_sum_3340[sample_3340,]
data_train_3340 = data_sum_3340[-sample_3340,]
#### 4417 ####
data_sum_4417 = data_sum %>% 
  filter(Industrial_sector == "4417") 
#sampling
sample_4417 = sample(length(data_sum_4417$HWsum), 0.2*length(data_sum_4417$HWsum))
data_test_4417 = data_sum_4417[sample_4417,]
data_train_4417 = data_sum_4417[-sample_4417,]
#### 4419 ####
data_sum_4419 = data_sum %>% 
  filter(Industrial_sector == "4419") 
#sampling
sample_4419 = sample(length(data_sum_4419$HWsum), 0.2*length(data_sum_4419$HWsum))
data_test_4419 = data_sum_4419[sample_4419,]
data_train_4419 = data_sum_4419[-sample_4419,]
#### 3120 ####
data_sum_3120 = data_sum %>% 
  filter(Industrial_sector == "3120") 
#sampling
sample_3120 = sample(length(data_sum_3120$HWsum), 0.2*length(data_sum_3120$HWsum))
data_test_3120 = data_sum_3120[sample_3120,]
data_train_3120 = data_sum_3120[-sample_3120,]


# Data consolidation ------------------------------------------------------
data_test = rbind(data_test_3360, data_test_2614,
                  data_test_2631, data_test_3130,
                  data_test_3982, data_test_2669,
                  data_test_3340, data_test_4417,
                  data_test_4419, data_test_3120)

data_train_pre = rbind(data_train_3360, data_train_2614,
                  data_train_2631, data_train_3130,
                  data_train_3982, data_train_2669,
                  data_train_3340, data_train_4417,
                  data_train_4419, data_train_3120)
write.csv(data_test, "test_data.csv")
write.csv(data_train_pre, "train_pre_data.csv")
