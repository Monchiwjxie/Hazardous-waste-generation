library(readr)
library(data.table)
library(DMwR2)                 
library(performanceEstimation)
library(UBL)
library(earth)
library(tidyverse)
library(operators)
library(fields)
library(ROCR)
library(gdata)
library(Hmisc)
library(kknn)
library(paradox)
library(mlr3)
library(mlr3verse)


# DIBS_function -----------------------------------------------------------
SMO_data <- function(dat, var_y, thr.rel, k, 
                     C.perc = "balance", repl = FALSE, 
                     dist = "HEOM", p = 2){
  #predecessor function
  DIBSRegress.exs <- function(orig, ind, tgt, N, k, dist, p, pert){
    indpos <- match(ind, rownames(orig))
    dat <- orig[indpos,]
    
    ConstFeat <- which(apply(dat, 2, function(col){length(unique(col)) == 1}))
    
    if(length(ConstFeat)){
      badds <- dat
      ConstRes <- dat[1,ConstFeat]
      dat <- dat[,apply(dat, 2, function(col) { length(unique(col)) > 1 })]
      tgt <- ncol(dat)
    }
    
    nomatr <- c()
    T <- matrix(nrow = dim(dat)[1], ncol = dim(dat)[2])
    for (col in seq.int(dim(T)[2])){
      if (class(dat[, col]) %in% c('factor', 'character')) {
        T[, col] <- as.integer(dat[, col])
        nomatr <- c(nomatr, col)
      } else {
        T[, col] <- dat[, col]
      }
    }
    nC <- dim(T)[2]
    nT <- dim(T)[1]
    
    
    ranges <- rep(1, nC)
    if (length(nomatr)) {
      for (x in (1:nC)[-c(nomatr)]) {
        ranges[x] <- max(T[, x]) - min(T[, x])
      }
    } else {
      for(x in (1:nC)) {
        ranges[x] <- max(T[, x]) - min(T[, x])
      }
    }
    
    # test that k is possible to use!
    # if(nrow(dat)<k+1){ 
    #   warning(paste("Unable to compute", k,"neighbours in this bump. Using",
    #                 nrow(dat)-1, "for kNN computation."), call.=FALSE)
    kNNs <- neighbours(tgt, dat, dist, p, k)
    DM <- distances(tgt, dat, dist, p)
    maxDM <- apply(DM, 1, function(x){ # half the median of the distances in the line
      summary(x)[3]/2
    })
    
    nexs <- as.integer(N - 1) # nr of examples to generate for each rare case
    extra <- as.integer(nT * (N - 1 - nexs)) # the extra examples to generate
    idx <- sample(1:nT, extra)
    newM <- matrix(nrow = nexs * nT + extra, ncol = nC)    # the new cases
    
    if (nexs) {
      for (i in 1:nT) {
        Slist <- which(DM[i,kNNs[i,]]<maxDM[i])
        for (n in 1:nexs) {
          # select randomly one of the k NNs
          neig <- sample(1:k, 1)
          if (neig %in% Slist){  ###### use SmoteR
            # the attribute values of the generated case
            difs <- T[kNNs[i, neig], -tgt] - T[i, -tgt]
            newM[(i - 1) * nexs + n, -tgt] <- T[i, -tgt] + runif(1) * difs
            for (a in nomatr) {
              # nominal attributes are randomly selected among the existing
              # values of seed and the selected neighbour 
              newM[(i - 1) * nexs + n, a] <- c(T[kNNs[i, neig], a],
                                               T[i, a])[1 + round(runif(1), 0)]
            }
            # now the target value (weighted (by inverse distance) average)
            d1 <- d2 <- 0
            for (x in (1:nC)[-c(nomatr, tgt)]) {
              d1 <- abs(T[i, x] - newM[(i - 1) * nexs + n, x])/ranges[x]
              d2 <- abs(T[kNNs[i, neig], x] - newM[(i - 1) * nexs + n, x])/ranges[x]
            }
            if (length(nomatr)) {
              d1 <- d1 + sum(T[i, nomatr] != newM[(i - 1) * nexs + n, nomatr])
              d2 <- d2 + 
                sum(T[kNNs[i, neig], nomatr] != newM[(i - 1) * nexs + n, nomatr])
            }
            # (d2+d1-d1 = d2 and d2+d1-d2 = d1) the more distant the less weight
            if (d1 == d2) {
              newM[(i - 1) * nexs + n, tgt] <- (T[i, tgt] + T[kNNs[i, neig], tgt])/2
            } else {
              newM[(i - 1) * nexs + n, tgt] <- (d2 * T[i, tgt] + 
                                                  d1 * T[kNNs[i, neig], tgt])/(d1 + d2)
            }
          } else { ####### use GaussNoise
            if(maxDM[i]>0.02){
              tpert <- 0.02
            } else {
              tpert <- maxDM[i]
            }
            id.ex <- (i - 1) * nexs + n 
            for (num in 1:nC) {
              if (is.na(T[i, num])) {
                newM[id.ex, num] <- NA
              } else {
                newM[id.ex, num] <- T[i, num] + rnorm(1, 0, sd(T[, num], 
                                                               na.rm = TRUE) * tpert)
                if (num %in% nomatr) {
                  probs <- c()
                  if (length(unique(T[, num])) == 1) {
                    newM[id.ex, num] <- T[1, num]
                  } else {
                    for (u in 1:length(unique(T[, num]))) {
                      probs <- c(probs, 
                                 length(which(T[, num] == unique(T[, num])[u])))
                    }
                    newM[id.ex, num] <- sample(unique(T[, num]), 1, prob = probs)
                  }
                }
              }
            }
            
          }
        }
      }
    }
    
    if (extra) {
      count <- 1
      for (i in idx) {
        Slist <- which(DM[i,kNNs[i,]]<maxDM[i])
        # select randomly one of the k NNs
        neig <- sample(1:k, 1) 
        if (neig %in% Slist){  ###### use SmoteR
          # the attribute values of the generated case
          difs <- T[kNNs[i, neig], -tgt] - T[i, -tgt]
          newM[nexs * nT + count, -tgt] <- T[i, -tgt] + runif(1) * difs
          for (a in nomatr) {
            newM[nexs * nT + count, a] <- c(T[kNNs[i,neig], a], 
                                            T[i, a])[1 + round(runif(1), 0)]
          }
          
          # now the target value (weighted (by inverse distance) average)
          d1 <- d2 <- 0
          for (x in (1:nC)[-c(nomatr,tgt)]) {
            d1 <- abs(T[i, x] - newM[nexs * nT + count, x])/ranges[x]
            d2 <- abs(T[kNNs[i, neig], x] - newM[nexs * nT + count, x])/ranges[x]
          }
          if (length(nomatr)) {
            d1 <- d1 + sum(T[i,nomatr] != newM[nexs *nT + count, nomatr])
            d2 <- d2 + 
              sum(T[kNNs[i, neig], nomatr] != newM[nexs * nT + count, nomatr])
          }
          # (d2+d1-d1 = d2 and d2+d1-d2 = d1) the more distant the less weight
          if (d1 == d2) {
            newM[nexs * nT + count, tgt] <- (T[i, tgt] + T[kNNs[i, neig], tgt])/2
          } else {
            newM[nexs * nT + count, tgt] <- (d2 * T[i, tgt] + 
                                               d1 * T[kNNs[i, neig],tgt])/(d1 + d2)
          }
        } else { ########### use GaussNoise
          if(maxDM[i]>0.02){
            tpert <- 0.02
          } else {
            tpert <- maxDM[i]
          }
          for (num in 1:nC) {
            if (is.na(T[i, num])) {
              newM[nexs * nT + count, num] <- NA
            } else {
              newM[nexs * nT + count, num] <- T[i, num] + rnorm(1, 0, sd(T[, num],
                                                                         na.rm = TRUE) * tpert)
              if (num %in% nomatr) {
                probs <- c()
                if (length(unique(T[, num])) == 1) {
                  newM[nexs * nT + count, num] <- T[1, num]
                } else {
                  for (u in 1:length(unique(T[, num]))) {
                    probs <- c(probs,
                               length(which(T[, num] == unique(T[, num])[u])))
                  }
                  newM[nexs * nT + count, num] <- sample(unique(T[, num]),
                                                         1, prob = probs)
                }
              }
            }
          }
        }
        count <- count + 1
      }
    }
    
    newCases <- data.frame(newM)
    for (a in nomatr) {
      newCases[, a] <- factor(newCases[, a],
                              levels = 1:nlevels(dat[, a]),
                              labels = levels(dat[, a]))
    }
    
    if(length(ConstFeat)){ # add constant features that were removed in the beginning
      
      newCases <- cbind(newCases, 
                        as.data.frame(lapply(ConstRes,
                                             function(x){rep(x, nrow(newCases))})))
      colnames(newCases) <- c(colnames(dat), names(ConstFeat))
      newCases <- newCases[colnames(badds)]
      
    } else {
      colnames(newCases) <- colnames(dat)
    }
    
    newCases
    
    
  }
  #Location of dependent variable
  tgt <- which(names(dat) == as.character(var_y))##Find the column where the dependent variable is located

  if (tgt < ncol(dat)) {
    orig.order <- colnames(dat)
    cols <- 1:ncol(dat)
    cols[c(tgt, ncol(dat))] <- cols[c(ncol(dat), tgt)]
    dat <- dat[, cols]
  }#The dependent variable is placed in the last column
  y <- dat[, ncol(dat)]
  attr(y, "names") <- rownames(dat)
  s.y <- sort(y)
  
  method <- "extremes"
  extr.type = "both"
  pc <- phi.control(s.y, method = method, extr.type = extr.type)
  temp <- phi(s.y, pc)
  #Temp detection
  if (!length(which(temp < 1))) {
    stop("All the points have relevance 1.
         Please, redefine your relevance function!")
  }
  if (!length(which(temp > 0))) {
    stop("All the points have relevance 0.
         Please, redefine your relevance function!")
  }
  
  bumps <- c()
  for (i in 1:(length(y) - 1)) { 
    if ((temp[i] >= thr.rel && temp[i+1] < thr.rel) || 
        (temp[i] < thr.rel && temp[i+1] >= thr.rel)) {
      bumps <- c(bumps, i)
    }
  }
  
  nbump <- length(bumps) + 1 # number of different "classes"
  obs.ind <- as.list(rep(NA, nbump))
  last <- 1
  for (i in 1:length(bumps)) {
    obs.ind[[i]] <- s.y[last:bumps[i]]
    last <- bumps[i] + 1
  }
  obs.ind[[nbump]] <- s.y[last:length(s.y)]
  
  newdata <- data.frame()
  
  if (is.list(C.perc)) {
    if (length(C.perc) != nbump){
      stop("The percentages provided must be the same length as the number
           of bumps!")
    }
  } else if (C.perc == "balance") {
    # estimate the percentages of over/under sampling
    B <- round(nrow(dat)/nbump, 0)
    C.perc <- B/sapply(obs.ind, length)        
  } else if (C.perc == "extreme") {
    B <- round(nrow(dat)/nbump, 0)
    rescale <- nbump * B/sum(B^2/sapply(obs.ind, length))
    obj <- round((B^2/sapply(obs.ind, length)) * rescale, 2)
    C.perc <- round(obj/sapply(obs.ind, length), 1)
  }
  
  for (i in 1:nbump) {
    if (C.perc[[i]] == 1) {
      newdata <- rbind(newdata, dat[names(obs.ind[[i]]), ])
    } else if (C.perc[[i]] > 1) {
      newExs <- DIBSRegress.exs(dat, names(obs.ind[[i]]), ncol(dat), C.perc[[i]],
                                k, dist, p, pert)
      # add original rare examples and synthetic generated examples
      newdata <- rbind(newdata, newExs, dat[names(obs.ind[[i]]), ])
    } else if (C.perc[[i]] < 1) {
      sel.maj <- sample(1:length(obs.ind[[i]]),
                        as.integer(C.perc[[i]] * length(obs.ind[[i]])),
                        replace = repl)
      newdata <- rbind(newdata, dat[names(obs.ind[[i]][sel.maj]), ])
    }
  }
  if (tgt < ncol(dat)) {
    newdata <- newdata[, cols]
    dat <- dat[, cols]
  }
  newdata
}

# find_K_function ---------------------------------------------------------
find_K <- function(dat, var_y){
  tgt <- which(names(dat) == as.character(var_y))
  if (tgt < ncol(dat)) {
    orig.order <- colnames(dat)
    cols <- 1:ncol(dat)
    cols[c(tgt, ncol(dat))] <- cols[c(ncol(dat), tgt)]
    dat <- dat[, cols]
  }
  
  y <- dat[, ncol(dat)]
  attr(y, "names") <- rownames(dat)
  s.y <- sort(y)
  
  ##Calculate the sparsity rate for different variables
  method <- "extremes"
  extr.type = "both"
  pc <- phi.control(s.y, method = method, extr.type = extr.type)
  temp <- phi(s.y, pc)
  
  ord <- order(dat[,tgt])
  sub_data <- dat[ord,]#
  sub_data <- sub_data[,-tgt]
  sub_data_std <- as.data.frame(scale(sub_data))
  sub_data_std$temp <- unlist(temp)#Add "temp" column
  
  knn <- lrn("regr.kknn", 
             kernel = "rectangular", scale = FALSE)
  task <- as_task_regr(sub_data_std, target = "temp")
  set.seed(4577)
  split = partition(task, ratio = 0.9)
  search_space = ps(
    k = p_int(lower = 2, upper = 15))
  knn.at = auto_tuner(
    learner = knn,
    resampling = rsmp("cv", folds = 10),
    measure = msr("regr.rmse"),
    search_space = search_space,
    method = "grid_search",
    term_evals = 10)
  knn.at$train(task, row_ids = split$train)
  a <- knn.at$tuning_result[["k"]]
  list(a)
}


# find_thr ----------------------------------------------------------------
dat = data_sum
var_y = "HWsum"
#### find thr function ####
tgt <- which(names(dat) == as.character(var_y))
if (tgt < ncol(dat)) {
  orig.order <- colnames(dat)
  cols <- 1:ncol(dat)
  cols[c(tgt, ncol(dat))] <- cols[c(ncol(dat), tgt)]
  dat <- dat[, cols]
}
y <- dat[, ncol(dat)]
attr(y, "names") <- rownames(dat)
s.y <- sort(y)

method <- "extremes"
extr.type = "both"
pc <- phi.control(s.y, method = method, extr.type = extr.type)
temp <- phi(s.y, pc)

#temp
if (!length(which(temp < 1))) {
  stop("All the points have relevance 1.
         Please, redefine your relevance function!")
}
if (!length(which(temp > 0))) {
  stop("All the points have relevance 0.
         Please, redefine your relevance function!")
}

tar = data.frame(sort(y), temp)

#### The temp value corresponding to the three-quarter quartile of the data is thr





