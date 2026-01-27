library(UBL)
data_train_smogn = SMOGNRegress(form = HWsum ~ .,
                                dat = data,
                                thr.rel = ,
                                dist = "HEOM",
                                k = ,
                                C.perc = "balance")
