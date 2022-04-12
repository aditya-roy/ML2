#====================================================================
#1 2017-2018
#====================================================================

library(mlbench)
library(glmnet)
data(Sonar)

N = nrow(Sonar)
P = ncol(Sonar)-1
M = 150 # size of training set
set.seed(1)
mdata = Sonar[sample(1:N),]
itrain = sample(1:N,M)
x = as.matrix(mdata[,-ncol(mdata)])
y = mdata$Class
xtrain = x[itrain,]
ytrain = y[itrain]
cs = data.frame(xtrain,ytrain)
#====================================================================
#1
#====================================================================

#how many observations in the test set
nrow(mdata[-itrain])
#208

#====================================================================
#2 Forward step wise selection
#Plot the BIC obtained as a function of model size
#====================================================================
reg.fwd = regsubsets(ytrain~., data=cs, nvmax=P, method="forward")
par(mfrow=c(1,1))

plot(summary(reg.fwd)$bic, t='b',xlab ='model size',ylab='BIC', pch=20, cex=1.5)
#
i.opt = which.min(summary(reg.fwd)$bic)
#how many variables does the model have
i.set.opt = summary(reg.fwd)$which[i.opt,]
summary(reg.fwd)$which[i.opt, i.set.opt]
#8 models

#====================================================================
#4 optimis the LASSO
#====================================================================
lasso.opt = cv.glmnet(as.matrix(x[itrain,]),y[itrain],alpha=1,family="binomial")
lambda=lasso.opt$lambda.min 
#0.02303016

#====================================================================
#5 fit lasso model
#====================================================================

lasso = glmnet(as.matrix(x[itrain,]),y[itrain],alpha=1,family="binomial",lambda=lasso.opt$lambda.min)

coefficients(lasso)

#====================================================================
#6 classification tree
#====================================================================
tree.model = tree(ytrain~., data=CS)
summary(tree.model)

#variables used in tree 
length(summary(tree.out)$used) #10

#====================================================================
#7 Random forest
#====================================================================
#Fitting a Random Forest
rf.out = randomForest(CS$ytrain~., CS)

#====================================================================
#8 Prediction from classification tree
#====================================================================
tree.pred = predict(tree.model,
                    newdata=as.data.frame(x[-itrain,]), type='class')


#All X value
#Confusion Matrix of tree
(tb1 = table(tree.pred,y[-itrain])) 

#====================================================================
#9 Prediction from classification tree using vectors
#====================================================================
tree.pred = predict(tree.model,
                    newdata=as.data.frame(x[-itrain,]), type='vector')

library(pROC)
#AUC and ROC 
#method 2 Using pROC::roc
a = pROC::roc(y[-itrain], tree.pred[,2], quiet = FALSE)
plot(a)

a$auc

rf.pred = predict(rf.out,x[-itrain,], type='prob')[,2]
roc.rf = roc(response=y[-itrain], predictor=rf.pred)
plot(roc.rf)
roc.rf$auc

#======================================================================
#====================================================================
#1 2019-2020
#====================================================================

library(leaps)
library(randomForest)
dat = read.csv(file="C:/Users/sairam/OneDrive/Desktop/ML Rcodes/Sem 2/CA -2/Q3_dataset.csv", stringsAsFactors=TRUE)
X = dat
X$Y <- NULL
Y = dat$Y

#====================================================
#a 10 fold CV Forward stepwise model selection
#====================================================
set.seed(1)
N = nrow(X)
P = ncol(X)
K = 10
folds = cut(1:N, K, labels=FALSE)
bics = matrix(NA, nrow=K, ncol=10)
vars = matrix(NA, nrow=K, ncol= P ) # fill in the blank for ncol
colnames(vars) = names(X)
for(k in 1:K){
  itr = which(folds!=k)
  
  reg.fwd = regsubsets(Y~., data=dat[itr,], nvmax=10, method="forward")
  
  bics[k,] = summary(reg.fwd)$bic
  vars[k,] = summary(reg.fwd)$which[which.min(bics[k,]),-1]
}

apply(vars,2,mean)*100

#================================================================
#B 10 fold CV Forward stepwise model selection, Optimal BIC Size
#================================================================
#3 is the most frequent mode size, meaning 3 variables are 
#being used

#================================================================
#C 10 fold CV Forward stepwise model selection, store BIC variables
#================================================================
#c
apply(bics,1,which.min) 
#================================================================
#D Useful variables using stepwise selection
#================================================================
#d
apply(vars,2,mean)*100
#feature 1,2,5 are selected 100 percent of times and feature 8 was 
#selected 30 percent of times
#================================================================
#E Random Forest Variable importance
#================================================================
#e
rf.out = randomForest(Y~.,data = X)
rf.out$mtry

rf.out$importance
names = colnames(X)
names[order(rf.out$importance,decreasing = TRUE)]

#f
#"X1"  "X2"  "X7"  "X5"  "X13"
#================================================================
#E Indicate why stepwise and RF variable selections differ
#================================================================
#g
#Stepwise selection is more adequate if the relationship between X and Y 
#is purely linear, and its output would be highly reliable(N>P)

#RF and linear regression would capture different aspects of the data
#if X-Y relationship is not linear; this would become apparent  in the
#selection of  "less important" predictors in the model

#RF was not cross validated hence the output variable importance assesments 
#is not adjusted for variability in the data.By applying CV we could end up 
#with an assessment of variable importance that would be comparable
#to that obtained from stepwise selection

#Choice of criterion : Stepwise selection was mad based on BIC, RF on the
#basis of decrease in MSE - so different creteria are applied to "rate" the
#covariates

#==============================================================
#2019 - 2020 final exam
#==============================================================
#==============================================================
#Question 3
#==============================================================
par(mfrow=c(1,1))
set.seed(4061)
x = read.csv(file="C:/Users/sairam/OneDrive/Desktop/ML Rcodes/Sem 2/Exam/Q3_x.csv", stringsAsFactors=TRUE)
y = read.csv(file="C:/Users/sairam/OneDrive/Desktop/ML Rcodes/Sem 2/Exam/Q3_y.csv", stringsAsFactors=TRUE)[,1]

x.valid = read.csv(file="C:/Users/sairam/OneDrive/Desktop/ML Rcodes/Sem 2/Exam/Q3_x_valid.csv", stringsAsFactors=TRUE)
y.valid = read.csv(file="C:/Users/sairam/OneDrive/Desktop/ML Rcodes/Sem 2/Exam/Q3_y_valid.csv", stringsAsFactors=TRUE)[,1]
#=============================================================
#a Random Forest variable importance
#=============================================================
CS = data.frame(x, y)

# grow a forest:
rf.out = randomForest(y~., CS)
#No. of variables tried at each split: 4

rf.v.imp = randomForest::varImpPlot(rf.out, pch=20)
# or:
# rf.v.imp = caret::varImp(rf.out)
# relative contribution of each feature:
sort(rf.v.imp[,1], decreasing=TRUE)/sum(rf.v.imp[,1])*100
rf.selection = row.names(rf.v.imp)[order(rf.v.imp[,1], decreasing=TRUE)]
# Top 5:
rf.selection[1:5]
#=============================================================
#b Bagging
#=============================================================
P = ncol(CS)-1

# compare to bagging:
bag.out = randomForest(y~., CS, mtry=P)

#fitting a variable importance graph
bag.v.imp = randomForest::varImpPlot(bag.out, pch=20)
# relative contribution of each feature:
sort(bag.v.imp[,1], decreasing=TRUE)/sum(bag.v.imp[,1])*100
bag.selection = row.names(bag.v.imp)[order(bag.v.imp[,1], decreasing=TRUE)]
# Top 5:
bag.selection[1:5]

#=============================================================
#c Variables selected by lasso
#=============================================================
library(glmnet)
set.seed(4061)
xm = model.matrix(y~.,data=x)[,-1]
class(xm)
lasso.lam = cv.glmnet(xm, y, alpha=1,family = "binomial") 

# quote optimal lambda:
lasso.lam$lambda.min
lasso.out = glmnet(xm,y,family="binomial",lambda=lasso.lam$lambda.min)
lasso.imp = abs(coef(lasso.out)[-1,])

# quote coefficients:
coef(lasso.out)
# relative contribution of each feature:
sort(lasso.imp, decreasing=TRUE)/sum(lasso.imp)*100
lasso.selection = names(lasso.imp)[order(lasso.imp, decreasing=TRUE)]
# Top 5:
lasso.selection[1:5]

#=============================================================
#d backward stepwise selection for a logistic regression model
#=============================================================

set.seed(4061)
glm.out = glm(y~., data=x, family="binomial")
step.out = step(glm.out, direction="backward")
summary(step.out)

coefs = abs(coef(step.out))/sum(abs(coef(step.out)), na.rm=TRUE)*100
coefs = coefs[-1]
coefs[order(abs(coefs), decreasing=TRUE)]

#=============================================================
#e Check correlation between variables
#=============================================================

M = round(cor(xm),3)
diag(M) = 0
M[which(abs(M) < .7)] = NA
M

#===========================================================
#Validation , lasso prediction,ROC Lasso
#===========================================================
# (6) validation:
# a) RF: confusion matrix and predict random forest
y.valid = as.factor(y.valid)
rf.p = predict(rf.out, newdata=x.valid)
rf.pr = predict(rf.out, newdata=x.valid, type="prob")[,2]
table(rf.p, y.valid)
# b) LASSO:confusion matrix and predict LASSO
xm.valid = model.matrix(y.valid~., data=x.valid)[,-1]
lasso.p = predict(lasso.out, newx=xm.valid, type="class")
lasso.pr = predict(lasso.out, newx=xm.valid)[,1]
table(lasso.p, y.valid)
# c) ROC analysis:for Lasso and rf
roc.rf = roc(response=y.valid, predictor=rf.pr)
roc.lasso = roc(response=y.valid, predictor=lasso.pr)
par(lwd=1)
plot(roc.rf, lty=1, lwd=3)
plot(roc.lasso, add=TRUE, col=2, lty=2, lwd=3)
legend("bottomright", bty="n", col=c(1,2), 
       legend=c("RF","LASSO"), lty=c(1,2), lwd=3)
roc.rf$auc
roc.lasso$auc
#============================================================
#Question 1 Boot strap samples  
#============================================================
library(e1071)
library(neuralnet)
library(tree)
library(pROC)
library(glmnet)
library(randomForest)
library(ISLR)
library(caret)
library(MASS)

# ----------------------------------------------------------------------
# Q2

# line up the data...

dat = MASS::Boston
N = nrow(dat)

# (2) OOB proportion:
#Distinct data points in boot strapping
B = 100
set.seed(4061)
sizes = numeric(B)
for(i in 1:B){ 
  x = sample(1:N,N,replace=TRUE)
  sizes[i]=length(unique(x)) 
}
summary(sizes/N)
boxplot(sizes/N)

# (3) 
#Boot strap for linear regression model
B = 100
set.seed(4061)
mse.lr = numeric(B)
for(b in 1:B){
  ib = sample(1:N, N, replace=TRUE)
  xb = dat[ib,]
  oob = dat[-unique(ib),]	
  lr = lm(medv~., data=xb)
  rrp = predict(lr, newdata=oob)
  mse.lr[b] = sqrt( mean( (rrp-oob$medv)^2 ) )
}
summary(mse.lr)
mean(mse.lr)

# (4) 
#Ridge regression in bootstrap
B = 100
set.seed(4061)
mse.rr = numeric(B)
for(b in 1:B){
  ib = sample(1:N, N, replace=TRUE)
  xb = dat[ib,]
  xbm = model.matrix(medv~., data=xb)[,-1]
  rr = glmnet(xbm, xb$medv, alpha=0, lambda=0.5)
  oob = dat[-unique(ib),]	
  oobm = model.matrix(medv~., data=oob)[,-1]
  rrp = predict(rr, newx=oobm)
  mse.rr[b] = sqrt( mean( (rrp-oob$medv)^2 ) )
}
mean(mse.rr)

boxplot(mse.lr, mse.rr)
abline(a=0, b=1)
#==============================================================
#2019 - 2020
#==============================================================

#Question 1
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

Y= dat$Days

#1
#Encoding is a required pre-processing step when working with categorical data 
#for machine learning algorithms.

#2

#a
par(mfrow=c(2,2), pch=20)
plot(factor(dat$Eth),Y, main="Days wrt\n Eth")
plot(factor(dat$Sex), Y, main="Days v\n Sex")
plot(factor(dat$Age), Y, main="Days v\n Age")
plot(factor(dat$Lrn), Y, main="Days v\n Lrn")

#b
#As a general rule, a variable will rank as more important if boxplots
#are not aligned horizontally.

#Statistical tests: percentiles are another used feature used by 
#them in order to determine -for example- if means across 
#groups are or not the same.

#Here ETH,SEX,AGE might be good predictors than Lrn
#as Lrn is symetriccaly alignment with Y and mean across the groups also seem same


#c

Days = dat$Days
Days[which(Days==0)] = 0.1
lDays = log(Days)
res1 <- t.test(as.numeric(dat$Eth),lDays)
res2 <- t.test(as.numeric(dat$Sex),lDays)
res3 <- t.test(as.numeric(dat$Age),lDays)
res4 <- t.test(as.numeric(dat$Lrn),lDays)
res1$p.value
res2$p.value
res3$p.value
res4$p.value

#pvalue = 3.831e-06 < 0.05, statistically significant difference in mean 

#d 
#log is used because data is skewed and to stabilise the residuals and make the data 
#normally distributed

#3
set.seed(4061)
itrain= sample(1:n, 100, replace=FALSE)
dat.train = dat[itrain,]
dat.test = dat[-itrain,]
x.train = dat.train[, 1:4]
y.train = dat.train$Days
x.test = dat.test[, 1:4]
y.test = dat.test$Days

features = data.frame(dat$Eth,dat$Sex,dat$Age,dat$Lrn)
set.seed(4061)
subsets <- c(1:4,ncol(features))
ctrl <- rfeControl(functions = rfFuncs,
                   method = "cv",
                   number = 10,
                   # method = "repeatedcv",
                   # repeats = 5,
                   verbose = FALSE)
rf.rfe <- rfe(x.train, y.train,
              sizes = subsets,
              rfeControl = ctrl)
rf.rfe

#b
#Sex, Age are important predictors according to rfe

#c
CS = data.frame(x.train, y.train)
CS.test = data.frame(x.test,y.test)

ytrain.rf =  randomForest(CS$y.train ~., CS)

#prediction
yhat.bag = predict(ytrain.rf,newdata = CS.test)

#MSE FOR TEST SET = 378.5773
mean((yhat.bag - y.test )^2)

#for two predictors

x.train1 = dat.train[, 2:3]
y.train1 = dat.train$Days
x.test1 = dat.test[, 2:3]
y.test1 = dat.test$Days

#c
CS1 = data.frame(x.train1, y.train1)
CS.test1 = data.frame(x.test1,y.test1)

ytrain.rf1 =  randomForest(CS1$y.train1 ~., CS1)

#prediction
yhat.bag1 = predict(ytrain.rf1,newdata = CS.test1)

#MSE FOR TEST SET = 457.0518
mean((yhat.bag1 - y.test1 )^2)
#======================================================================
#Question 2
#======================================================================

#1
#matrix of subset selection used in feature selection
#output 2 is from stepwise
#b
#Given a response vector Y ??? R
#n, predictor matrix X ??? Rn×p
#and a subset size k between 0 and
#min{n, p}, best subset selection finds the subset of k 
#predictors that produces the best fit in terms of squared error,

#in subset approach nvmax is number of attempts, usually we do it for whole 
#variable list

#lasso tends to shrink the coeff and remove that variable

#======================================================================
#Question 3
#======================================================================

rm(list=ls())
library(ISLR)
library(randomForest)
library(caret)
library(pROC)
library(leaps) # contains regsubsets()
library(glmnet)
set.seed(4061)
n = nrow(OJ)
dat = OJ[sample(1:n, n, replace=FALSE),]
set.seed(4061)
itrain = sample(1:n, round(.7*n), replace=FALSE)
dat.train = dat[itrain, ]
dat.test = dat[-itrain, ]
ncol(dat)

#==================================================================
#a
# stepwise  selections from step():
glm.out = glm(dat.train$Purchase~., data=dat.train, family="binomial")
set.seed(4061)
step.bth = step(glm.out, direction="both")

# NB: we can also assess feature contributions in terms 
# of magnitude of their effect:
coefs = abs(coef(step.bth))/sum(abs(coef(step.bth)), na.rm=TRUE)*100
coefs = coefs[-1]
coefs[order(abs(coefs), decreasing=TRUE)]
#===================================================================
#b
#variable seems most important
#PctDiscMM     DiscMM    LoyalCH  PctDiscCH    PriceMM    PriceCH    StoreID 


#===================================================================
#c Random Forest
#===================================================================
xtrain = dat.train
xtrain$Purchase = NULL
ytrain = dat.train$Purchase
CS = data.frame(xtrain, ytrain) 

#Fitting a Random Forest
rf.out = randomForest(CS$ytrain~., CS)

#fitting a variable importance graph
varImpPlot(rf.out, pch=15, main="Random Forest Variable importance")

#5 most important variables from RF
#LoyalCH,WeekofPurchase,StoreID,PriceDiff,SalePriceMM


#=======================================================================
#d Confidence interval
#======================================================================
glm.pred = (predict(glm.out, nedata = dat.train,
                    type = "response") > 0.5)
y.lm = ifelse(dat.train$Purchase == 'CH',FALSE, TRUE) 
tb = table(y.lm, glm.pred)
sum(diag(tb))/sum(tb)

rf.pred = predict(rf.out, newdata = dat.train,
                  type = "response")
confusionMatrix(dat.train$Purchase, rf.pred)$overall[1]
#===========================================================
#eConfidence interval
#===========================================================
library(pROC)
rf.pred = predict(rf.out, newdata = dat.test,
                  type = "prob")
roc(dat.test$Purchase, rf.pred[,1])$auc
rf.pred = predict(rf.out, newdata = dat.test,
                  type = "class")
confusionMatrix(dat.test$Purchase, rf.pred)$overall[1]
confusionMatrix(dat.test$Purchase, rf.pred)
# 95% CI : (0.7493, 0.8401)

glm.pred = predict(glm.out, newdata = dat.test,
                   type = "response")
roc(dat.test$Purchase, glm.pred)$auc

glm.pred = as.factor((predict(glm.out, newdata = dat.test,
                              type = "response")>0.5))
tb = table(dat.test$Purchase, glm.pred)
sum(diag(tb))/sum(tb)

y.lm = as.factor(ifelse(dat.test$Purchase == 'CH',FALSE, TRUE)) 
confusionMatrix(y.lm, glm.pred)
class(y.lm)

#========================================================
#a 2020-2021
#========================================================

library(MASS)
library(tree)
library(randomForest)
dat = Boston # NOTE: this is a data.frame...
set.seed(6041)
dat = dat[sample(1:nrow(dat)),]
dat$zn <- NULL
X = dat
X$medv <- NULL
X = scale(X) # NOTE: this makes it a matrix...
Y = dat$medv

#========================================================
#a
#========================================================
ncol(X)
#12
#B model coefficients in nueral network

#M *(P+1)+(M+1)K
#m = hidden layer nuerons
#P = input layer
#k = output layer 
#========================================================
#b correlation betwen variables
#========================================================
corr = cor(X)

corr[1:3,1:3]

        #crim      indus        chas
#crim   1.00000000 0.40658341 -0.05589158
#indus  0.40658341 1.00000000  0.06293803
#chas  -0.05589158 0.06293803  1.00000000
#========================================================
#c Statastical summaries of distribution
#========================================================
set.seed(6041)
itrain <- 1:400
xtrain <- X[itrain,]
xtest <- X[-itrain,]
ytrain <- Y[itrain]
ytest <- Y[-itrain]
set.seed(6041)
par(mfrow=c(2,2))
hist(ytrain)
hist(ytest)
boxplot(ytrain)
boxplot(ytest)
par(mfrow=c(1,1))
#========================================================
#d 
#========================================================

CS = data.frame(xtrain, ytrain)
CS.test = data.frame(xtest,ytest)

ytrain.rf =  randomForest(CS$ytrain ~., CS)

#MSE OF TRAINING SET = 11.48018

#prediction
yhat.bag = predict(ytrain.rf,newdata = CS.test)

#MSE FOR TEST SET = 10.37689
mean((yhat.bag - ytest )^2)

rf.oxy = randomForest(x=xtrain, y=ytrain,
                      xtest=xtest, ytest=ytest)
#=====================================================================
#2017-2018 final paper
#=====================================================================
class(iris)
library(tree)
library(glmnet)
library(pROC)
library(class)
library(randomForest)

M = 100
set.seed(4061)
dat = iris[sample(1:nrow(iris)),]
dat[,1:4] = apply(dat[,1:4],2,scale)
itrain = sample(1:nrow(iris), M)
x = dat[,-ncol(dat)]
y = dat$Species
xtrain = dat[itrain,-ncol(dat)]
ytrain = dat[itrain,]$Species
#=====================================================================
#Question 1
#=====================================================================
#a
#taking training data as dataframe
CS = data.frame(xtrain, ytrain) 
CSPred = data.frame(x, y)
CS.train = CSPred[itrain,] #taking training data
CS.test = CSPred[-itrain,] #Taking Test Data
CS.test1 = CSPred[-itrain,]$y #Taking Y values of Test data
CS.trainy = CSPred[itrain,]$y 

tree.out = tree(ytrain~., CS, split = c("deviance", "gini")) #regressing Y training with whole Train data
summary(tree.out)
length(summary(tree.out)$used)
# plot the tree
plot(tree.out)
text(tree.out, pretty=0)


#Variables are found useful
summary(tree.out)$used

#number of terminal nodes 
#6

#the misclassification error rate for the full tree
#tree.out tree function fit here,Here CS.test = test data
tree.pred = predict(tree.out, CS, type="class")

#All X value
#Confusion Matrix of tree
(tb1 = table(tree.pred,CS.trainy)) 

#Classification of error rate in trees
1- sum(diag(tb1))/sum(tb1)

#c
#generate the box plot for the predictors
par(mfrow=c(2,2))
# here's just a loop to save having to write 4 boxplot
# instructions with names by hand (being lazy often 
# makes for nicer code):
for(j in 1:4){ 
  boxplot(dat[,j]~dat$Species,
          xlab = 'Species',
          ylab = 'predictor',
          col=c('cyan','pink','red'), 
          main=names(dat)[j])
}

#e #predicions on test set
tree.pred.test = predict(tree.out, CS.test, type="class")

#Confusion Matrix of tree
(tb.test = table(tree.pred.test,CS.test1)) 

#Classification of error rate in trees
1- sum(diag(tb.test))/sum(tb.test)

#f (pruning)
cv.CS = cv.tree(tree.out, FUN=prune.misclass)
opt.size = cv.CS$size[which.min(cv.CS$dev)]

ptree = prune.misclass(tree.out, best=opt.size)
ptree 
summary(ptree)
par(mfrow=c(1,2))
plot(tree.out)
text(tree.out, pretty=0)
plot(ptree)
text(ptree, pretty=0)

#What is the optimal size for tree pruning??
#6
#CV reports more realistic deviance than the unprund tree

#g
#Fitting a Random Forest

rf.out = randomForest(CS$ytrain~., CS)

# matrix for the OOB observations:
(tb.rf2 = rf.out$confusion)

#error rate
1-sum(diag(tb.rf2))/sum(tb.rf2)

#h
# fitted values for "test set"
rf.yhat = predict(rf.out, CS.test, type="class")
# confusion matrix for RF (test data):
(tb.rf = table(rf.yhat, CS.test$y))

#Classification error rate
1 - sum(diag(tb.rf))/sum(tb.rf)

#i
#Random Forest ROC
#ROC curves are typically used in binary classification to study 
#the output of a classifier. In order to extend ROC curve and ROC 
#area to multi-label classification, it is necessary to binarize the output.

#=======================================================================
#Question 2
#=======================================================================
class(College)
dat = model.matrix(Apps~., College)[,-1]
dat <- apply(dat,2,scale)
set.seed(4061)
itrain = sample(1:nrow(dat), 500)
x = dat
y = College$Apps

xtest = dat[-itrain,]
ytest = College[-itrain,]$Apps

xtrain = dat[itrain,]
ytrain = College[itrain,]$Apps

CS.Train.2 = data.frame(xtrain,ytrain)
CS.Test.2 = data.frame(xtest,ytest)


lasso.cv = cv.glmnet(xtrain, ytrain, alpha=1,family = "gaussian")
lambda=lasso.cv$lambda.min 

lasso = glmnet(xtrain, ytrain, alpha=1, lambda=lasso.cv$lambda.min,family = "gaussian")

coefficients(lasso)

lasso.pred.test = predict(lasso, newx=xtest)


# MSE for the predicted data:
sqrt(mean((lasso.pred.test-ytest)^2))

#Correlation
cor(lasso.pred.test,ytest)

#Plot of Predicted VS True Test
plot(lasso.pred.test,ytest)

#d Random forest for split of 5
class(College)
dat = College
x1 = College
x1$Apps = NULL
#x1 <- apply(x1,2,scale)
set.seed(4061)
itrain = sample(1:nrow(dat), 500)
y = College$Apps

xtest = x1[-itrain,]
ytest = y[-itrain]

xtrain = x1[itrain,]
ytrain = y[itrain]

CS.Train.2 = data.frame(xtrain,ytrain)
CS.Test.2 = data.frame(xtest,ytest)

rf.mod5 = randomForest(ytrain~.,CS.Train.2,mtry = 5)
rf.mod15 = randomForest(ytrain~.,CS.Train.2,mtry = 15)

# fitted values for "train set"
rf.mod15.pred = predict(rf.mod15, CS.Train.2, type="response")

# confusion matrix for RF (test data):
(tb.rf = table(rf.mod15.pred, CS.Train.2$ytrain))

1-sum(diag(tb.rf))/sum(tb.rf)

#MSE OF TRAINING SET = 702.9414
sqrt(mean((rf.mod15.pred - ytrain )^2))

#prediction
yhat.test = predict(rf.mod15, CS.Test.2, type="response")

#MSE FOR TEST SET = 1032.442
sqrt(mean((yhat.test - ytest )^2))

#==========================================================================
#Question 3
#==========================================================================
class(Smarket)
x = Smarket[,-9]
y = Smarket$Direction
set.seed(4061)
train = sample(1:nrow(Smarket),1000)
xtrain = x[train,]
ytrain = y[train]
xtest  = x[-train,]
ytest = y[-train]
CS.test = data.frame(xtest, ytest) 


#Taking only 250 observations of test data
xtest.250 = x[itest,]
ytest.250 = y[itest]
CS.test1 = data.frame(xtest.250, ytest.250) 
#a
#Fitting a Random Forest
rf.out1 = randomForest(CS$ytrain~., CS)

# fitted values for "train set"
rf.yhat1 = predict(rf.out1, CS, type="class")

# confusion matrix for RF (train data):
(tb.rf1 = table(rf.yhat1, CS$ytrain))

#Classification error rate
1 - sum(diag(tb.rf1))/sum(tb.rf1)

#b
# fitted values for "test set"
rf.yhat2 = predict(rf.out1, CS.test, type="class")

# confusion matrix for RF (test data):
(tb.rf2 = table(rf.yhat2, CS.test$ytest))

#Random Forest ROC
rf.p1 = predict(rf.out1, CS.test, type="prob")[,2] 
roc.rf = roc(response=CS.test$ytest, predictor=rf.p1)
roc.rf$auc
plot(roc.rf)


#c

x = Smarket[,-9]
y = Smarket$Direction
set.seed(4061)
train = sample(1:nrow(Smarket),1000)


x.train = x[train,]
y.train = y[train]

x.test = x[-train,]
y.test = y[-train]

ko = knn(x.train, x.test, y.train, k=2, prob=TRUE)
tb = table(ko, y.test)
1 - sum(diag(tb)) / sum(tb)

#d
Kmax = 10
acc = numeric(Kmax)
for(k in 1:Kmax){
  ko = knn(x.train, x.test, y.train, k)
  tb = table(ko, y.test)
  acc[k] = sum(diag(tb)) / sum(tb)	
}
plot(1-acc, pch=20, t='b', xlab='k')

#=====================================================================
#2020-2021 final paper
#=====================================================================
#Question 2
library(gbm)
library(caret)
dat = read.csv(file="C:/Users/sairam/OneDrive/Desktop/ML Rcodes/Sem 2/CA - 1/dodgysales.csv", stringsAsFactors=TRUE)
n = nrow(dat)
set.seed(6041)
i.train = sample(1:n, floor(.7*n))
dat.train = dat[i.train,]
dat.validation = dat[-i.train,]

#=============================================================================
#a
#=============================================================================

#Regression Problem
#=============================================================================
#b
#=============================================================================
ncol(dat.train)-1

#12
#=============================================================================
#c(i)
#=============================================================================
# re-encode method dataset
myrecode <- function(x){
  # function recoding levels into numerical values
  if(is.factor(x)){
    levels(x)
    return(as.numeric(x)) 
  } else {
    return(x)
  }
}

#scale dataset
myscale <- function(x){
  # function applying normalization to [0,1] scale
  minx = min(x,na.rm=TRUE)
  maxx = max(x,na.rm=TRUE)
  return((x-minx)/(maxx-minx))
}

#applying the encoding
dat.s = data.frame(lapply(dat,myrecode))

#applying the scaling
dat.s = data.frame(lapply(dat.s,myscale))

#summary of dat.s$Sales,dat.s$BudgOp

summary(dat.s$Sales)
summary(dat.s$BudgOp)

#Frequency distribution table

my_tab <- table(dat.s$Training) 

prop.table(table(dat.s$Training))
plot(as.factor(dat.s$Training))

#=============================================================================
#c(ii) Train single Nueral network
#=============================================================================
library(neuralnet)
dat.s.train = dat.s[i.train,]
dat.s.validation = dat.s[-i.train,]

nno1 = nnet(dat.s.train$Sales~., data=dat.s.train, size=c(3),decay=c(0), linout=1)
nno2 = nnet(dat.s.train$Sales~., data=dat.s.train, size=c(8),decay=c(0), linout=1)

#MSE train for Nueral Network 1
nno1mse = mean((nno1$fitted.values -dat.s.train$Sales)^2)

#MSE train for Nueral Network 2
nno2mse = mean((nno2$fitted.values -dat.s.train$Sales)^2)
nno1mse
nno2mse
#=============================================================================
#c(iii) Predict single Nueral network
#=============================================================================

#pred for NN1

preds = predict(nno1, newdata=dat.s.validation)

mean((preds-dat.s.validation$Sales)^2)

#pred for NN2

preds = predict(nno2, newdata=dat.s.validation)

mean((preds-dat.s.validation$Sales)^2)

#=============================================================================
#Gradient boosting using caret for regression
#========================================================================

ytrue = dat.validation$Sales
yfitted = dat.train$Sales

#For regresssion give "Gaussian"

#gb.out = train(y.train ~., data=CS.train, method='gbm', distribution='gaussian')
gb.out = train(Sales~., data=dat.train, method='gbm', distribution='bernoulli',n.trees = 100)


#MSE for Train
gb.fitted = predict(gb.out) # corresponding fitted values
mean((gb.fitted-yfitted)^2)

#MSE for Test
gb.pred = predict(gb.out, dat.validation)
mean((gb.pred-ytrue)^2)
#=================================================================
#GLM Family is gaussian if its regression

fit = glm(Sales~., data=dat.train, family="gaussian")
pred = predict(fit, newdata=dat.validation, type="response")

#MSE for Train
mean((fit$fitted.values - dat.train$Sales)^2)

#MSE for Test
mean((pred- dat.validation$Sales)^2)

#==================================================================
#Ridge Regression alpha = 0 , lasso, alpha = 1

library(glmnet)
set.seed(6041)

x = model.matrix(Sales~., data=dat)[,-1]
y = dat$Sales
x.train = x[i.train,]
x.test = x[-i.train,]
y.train = y[i.train]
y.test = y[-i.train]

# set alpha=0 for ridge regression
lamda.cv = cv.glmnet(x.train, y.train, alpha=0)

lamda.cv$lambda.min

ridge.mod.cv = glmnet(x.train, y.train, alpha=0, lambda=lamda.cv$lambda.min)
ridge.pred.train = predict(ridge.mod.cv, newx=x.train)

#MSE for the training data:
mean((ridge.pred.train-y.train)^2)
ridge.pred.test = predict(ridge.mod.cv, newx=x.test)

# MSE for the test data:
mean((ridge.pred.test-y.test)^2)
#=============================================================================
#h Feature elimination rfe for random forest
#=============================================================================
library(gbm)
library(caret)
dat = read.csv(file="C:/Users/sairam/OneDrive/Desktop/ML Rcodes/Sem 2/CA - 1/dodgysales.csv", stringsAsFactors=TRUE)
n = nrow(dat)
set.seed(6041)
i.train = sample(1:n, floor(.7*n))
dat.train = dat[i.train,]
dat.validation = dat[-i.train,]
ncol(dat.train)
set.seed(4061)
subsets <- c(1:5, 10, 15, 20, ncol(ncol(dat.train)))
ctrl <- rfeControl(functions = rfFuncs,
                   method = "cv",
                   number = 10,
                   # method = "repeatedcv",
                   # repeats = 5,
                   verbose = FALSE)
rf.rfe <- rfe(x.train, y.train,
              sizes = subsets,
              rfeControl = ctrl)
rf.rfe

#=============================================================================
#i Feature elimination rfe based on backward step elimination
#=============================================================================

#=============================================================================
#Question 3 Regular Cross validation
#=============================================================================

library(mlbench)
library(e1071)
data(BreastCancer)
dat = na.omit(BreastCancer)
dat$Id = NULL
set.seed(4061)
i.train = sample(1:nrow(dat), 600, replace=FALSE)
dat.train = dat[i.train,]
dat.validation = dat[-i.train,]

K = 10
N = length(i.cv)
folds = cut(1:N, K, labels=FALSE)
acc.rf = acc.svml = acc.svmr= numeric(K)
split.rf = numeric(k)
for(k in 1:K){
  itrain	= which(folds!=k)
  dattrain = dat.train[itrain, ]
  dattest = dat.train[-itrain, ]
  
  # train classifiers:
  
  #Random forest
  rf.out = randomForest(Class~., data=dattrain)
  #number of splits used at each split
  split.rf[k] = rf.out$mtry
  
  #SVM Linear Kernel
  svmo.lin = svm(Class~., data=dattrain, kernel='linear', scale=F)
  
  #SVM Radial Kernel
  svmo.rad = svm(Class~., data=dattrain, kernel='radial', scale=F)
  
  #test classifiers
  
  #Random Forest
  rf.p = predict(rf.out, newdata=dattest)
  
  #SVM Linear Kernel
  svmp = predict(svmo.lin, newdata=dattest)
  
  #SVM Radial Kernel
  svmpr = predict(svmo.rad, newdata=dattest)
  
  # corresponding confusion matrices:
  #RF
  tb.rf = table(rf.p, dattest$Class)
  
  #SVML
  tb.svml = table(svmp,dattest$Class)
  
  #SVMR
  tb.svmr = table(svmpr,dattest$Class)
  
  # store prediction accuracies:
  
  #Random Forest
  acc.rf[k] = sum(diag(tb.rf)) / sum(tb.rf)
  
  #SVM Linear
  acc.svml[k] = sum(diag(tb.svml)) / sum(tb.svml)
  
  #SVM Radial
  acc.svmr[k] = sum(diag(tb.svmr)) / sum(tb.svmr)
  
  
}

#test set prediction accuracy for RF
mean(acc.rf)

#test set prediction accuracy for SVM Linear
mean(acc.svml)

#test set prediction accuracy for SVM Radial
mean(acc.svmr)

#Splits used at each step
split.rf

#=========================================================================
#using Caret CV for RF and SVM
#=========================================================================

library(mlbench)
data(BreastCancer)
library(randomForest)
library(pROC)
library(caret)
dat = na.omit(BreastCancer)
dat$Id = NULL
set.seed(4061)
i.train = sample(1:nrow(dat), 600, replace=FALSE)
dat.train = dat[i.train,]
dat.validation = dat[-i.train,]
x = dat.train
x$Class = NULL
y = as.factor(dat.train$Class)
# We can also refine/modify tuning:
train.control = trainControl(method='cv',
                             number=10,
                             savePredictions='final',
                             classProbs=TRUE,
                             summaryFunction=twoClassSummary)

#Random Forest
rf.out = caret::train(Class~., data=dat.train, method='rf', trControl=train.control, metric='ROC')
rf.fitted = predict(rf.out) # corresponding fitted values
# (2) Validation-set predictions on independent validation set
rf.pred = predict(rf.out, dat.validation)
# Associated performance evaluation
rf.cm = confusionMatrix(reference=dat.validation$Class, data=rf.pred,
                        mode="everything")
accuracy = rf.cm$overall[0:1]

### 3b: SVM (linear)
linear.out = caret::train(Class~., data=dat.train, method="svmLinear", trControl=train.control)
linear.pred = predict(linear.out, dat.validation)
linear.cm = confusionMatrix(reference=dat.validation$Class, data=linear.pred, mode="everything")
linear.cm

### 3c:SVM (radial)
Radial.out = caret::train(Class~., data=dat.train, method="svmRadial", trControl=train.control)
Radial.pred = predict(Radial.out, dat.validation)
Radial.cm = confusionMatrix(reference=dat.validation$Class, data=Radial.pred, mode="everything")

####3d
accuracy = rbind(rf.cm$overall[0:1], linear.cm$overall[0:1], Radial.cm$overall[0:1])
row.names(accuracy) = c("RF","SVM.linear","SVM.radial")
round(accuracy, 3)

###3e
rf.importance <- varImp(rf.out, scale=FALSE)
rf.importance

linear.importance <- varImp(linear.out, scale=FALSE)
linear.importance

radial.importance <- varImp(Radial.out, scale=FALSE)
radial.importance


