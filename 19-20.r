#Q1
#1. RF/NN (i think)

#2.
rm(list=ls()) 

library(MASS) 
library(randomForest)
library(nnet)
library(e1071)
library(gbm) 
library(caret)

set.seed(4061)
n = nrow(quine)
dat = quine[sample(1:n, n, replace=FALSE),]
Days = dat$Days
Days[which(Days==0)] = 0.1
lDays = log(Days)
par(mfrow=c(2,2))

#2.a
for(i in 1:4){
  boxplot(dat$Days~dat[,i],main=names(dat)[i],col=2:4)
}
#lmo = lm(Days~., data=dat)
#plot(lmo$residual)
#plot(lmo$fitted.values, lmo$residuals, pch=20)
#hist(lmo$residuals)
#qqnorm(lmo$residuals)
#qqline(lmo$residuals, col=2)

#2b. no clue what to explain from the plots

#2.c
dat2 = data.frame(dat,lDays)
lmo2 = lm(lDays~., data=dat2)
summary(lmo2)

#2d. cause some of days has 0 (i think)


#3
set.seed(4061)
itrain= sample(1:n, 100, replace=FALSE)
dat.train = dat[itrain,]
dat.test = dat[-itrain,]
x.train = dat.train[, 1:4]
y.train = dat.train$Days
x.test = dat.test[, 1:4]
y.test = dat.test$Days

subsets <- c(1:10, ncol(dat))
ctrl <- rfeControl(functions = rfFuncs,
                   method = "cv",
                   number = 10,
                   # method = "repeatedcv",
                   # repeats = 5,
                   verbose = FALSE)
set.seed(4061)
rf.rfe <- rfe(x.train, y.train,
              sizes = subsets,
              rfeControl = ctrl)
rf.rfe
#3a. Sex, Age

#3b. Age is significant

#3c. rmse
rfe.pred = predict(rf.rfe, x.test)
cbind(y.test, rfe.pred)

rf.all = randomForest(Days~., data=dat.train)
rf.predall = predict(rf.all, newdata=dat.test, type='class')
rf.2 = randomForest(Days~Sex+Age,data=dat.train)
rf.pred2 = predict(rf.2, newdata=dat.test)
sqrt( mean( (y.test-rf.predall)^2))
sqrt( mean( (y.test-rf.pred2)^2))

#=============================
#Q2

#1a. Output1: step(), Output2: lasso/ridge

#1b. similarity: they both mute useless predictors
# Diff: not sure

#2a. Yi=1, pi>0.5 ie, p in range of [0.5,1]
# Yi=0, pi between [0,1]
# cause sum(pi)/n < 0.5
# (3.85+unknown)/10 < 0.5 => unknown < 1.25, so can be value from [0,1]

#--------------------------------
#Q3
rm(list=ls())
library(ISLR) 
library(randomForest)
library(caret)
library(pROC)
set.seed(4061)
n = nrow(OJ)
dat = OJ[sample(1:n, n, replace=FALSE),]
set.seed(4061)
itrain = sample(1:n, round(.7*n), replace=FALSE) 
dat.train = dat[itrain, ]
dat.test = dat[-itrain, ]

#1. step()
library(stats)
# stepwise  selections from step():
lm.out = lm(as.numeric(Purchase)~., data=dat.train) 
set.seed(4061)
step.bth = step(lm.out, direction="both")
summary(step.bth)

#2. LoyalCH as its most significant

#3.rf
rf.fit = randomForest(Purchase~., data=dat.train)
length(rf.fit)
varImpPlot(rf.fit)
#Step: PriceMM,LocyalCH,PriceCH,DiscMM,PctDiscCH
#RF: LoyalCH, WeekofPur,StoreID,PriceDiff,SalePriceMM,ListPriceDiff

#4.
rf.pred = predict(rf.fit, newdata=dat.test, type='class')
confusionMatrix(reference=dat.test$Purchase, data=rf.pred,mode='everything')
#95% CI : (0.7594, 0.8486)
