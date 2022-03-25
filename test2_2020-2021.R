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

#(a)
ncol(X)-1
#11


#b
corr = cor(X)
corr[1:3,1:3]

        #crim      indus        chas
#crim   1.00000000 0.40658341 -0.05589158
#indus  0.40658341 1.00000000  0.06293803
#chas  -0.05589158 0.06293803  1.00000000

#(c)

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

#d
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
