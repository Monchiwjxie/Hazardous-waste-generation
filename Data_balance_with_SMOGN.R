library(UBL)
data_train_smogn = SMOGNRegress(form = HWsum ~ .,
                                dat = data_train,
                                thr.rel = ,
                                dist = "HEOM",
                                k = ,
                                C.perc = "balance")
