# 01 EXERCISE ------------------------------------------

set.seed(4060)

rm(list = ls())

# Loading data

library(ISLR)

library(glmnet)

data = Hitters

data = na.omit(data)

class(data$League)

# PART 2) -----------------------------------------------------------------

# Grid search for tunning lasso penalisation

grid = 10^seq(10,-2,length=100)

# LASSO

a = sapply(data, class)

b = a[a == "factor"]

d = names(b)

# We have to drop non.numerical variables

data = data[,-which(names(data)%in%d)]

# K-fold cross validation

K = 10

n = nrow(data)

folds = cut(1:n, K, labels = F)

storage = storager = matrix(NA, nrow = 100, ncol = K)

# loop

for(i in 1:100){
  
  # Shuffle data
  
  data = data[sample(1:n, n, replace = F), ]
  
  for(k in 1:K){
    
    train = which(folds != k)
    
    y = data$Salary
    
    xm = as.matrix(data[,-17])
    
    lasso = glmnet(xm[train,], y[train] ,
                   lambda = grid[i],
                   alpha = 1)
    
    # Validation errors
    
    yh.p = predict(lasso, newx = xm[-train,])
    
    storage[i,k] = sqrt(mean( (y[-train] - yh.p)^2 ))
    
    # Ridge
    
    ridge = glmnet(xm[train,], y[train] ,
                   lambda = grid[i],
                   alpha = 0)
    
    # Validation errors
    
    yh.r = predict(ridge, newx = xm[-train,])
    
    storager[i,k] = sqrt(mean( (y[-train] - yh.r)^2 ))
  }
}

# Select the best parameter lambda

# Lasso

mean.vector.l = apply(storage, 1, mean)

i.best = which.min(mean.vector.l)

lambda.bestlas = grid[i.best]

mean(mean.vector.l)

# Ridge

mean.vector.r = apply(storager, 1, mean)

i.best = which.min(mean.vector.r)

lambda.bestrid = grid[i.best]

mean(mean.vector.r)


# Ridge seems to be that works better....


