library(mlbench) 
data(Sonar) 
N = nrow(Sonar) 
P = ncol(Sonar)-1 
M = 150  # size of training set 
set.seed(1) 
mdata = Sonar[sample(1:N),] 
itrain = sample(1:N,M) 
x = as.matrix(mdata[,-ncol(mdata)]) #don't use
y = mdata$Class  #don't use
x = mdata[,-ncol(mdata)]
y = mdata[,ncol(mdata)]

#1
N-M
#2
#Step-wise Forward selection
library(leaps)
fwd.model = regsubsets(y[itrain]~., data = x[itrain,], method="forward",nvmax=P)
names(summary(fwd.model))
plot(summary(fwd.model)$bic, pch=20, t='b', xlab='model size', ylab='BIC')

#3
#size of the optimal model
min.bic.ind = which.min (summary(fwd.model)$bic)
summary(fwd.model)$outmat[min.bic.ind,]
#Optimal model has 8 variables

#4
#CV-lasso
install.packages("cv")
library(glmnet)
lasso.opt = cv.glmnet(as.matrix(x[itrain,]),y[itrain], alpha=1, family='binomial')
lasso.opt$lambda.min


#5
lasso.mod = glmnet(as.matrix(x[itrain,]),y[itrain], lambda = lasso.opt$lambda.min, family='binomial')

#a
names(lasso.mod)

lasso.mod$beta

#b
sum(lasso.mod$beta!=0)
length(which(lasso.mod$beta!=0))

#6
tree.model = tree(y[itrain]~., data=(x[itrain,]))
#number of variables used
summary(tree.model)
summary(tree.model)$used
length(summary(tree.model)$used)

#7
rf.model = randomForest(y[itrain]~., data = x[itrain,])
varImpPlot(rf.model)

#8
tree.pred = predict(tree.model, newdata = as.data.frame(x[-itrain,]),type='class')
table(tree.pred, y[-itrain])

#9
library(pROC)
tree.pred = predict(tree.model, newdata = as.data.frame(x[-itrain,]),type='vector')
tree.roc = roc(response = y[-itrain],predictor = tree.pred[,1])
plot(tree.roc)
tree.roc$auc

rf.pred = predict(rf.model, x[-itrain,],type='prob')
rf.roc = roc(response=y[-itrain], predictor = rf.pred[,1])
rf.roc$auc

par(new=TRUE)
plot(rf.roc,col=4)
legend("bottomright", legend=c("Tree","RF"),col=c(1,4),pch=15,bty='n')





