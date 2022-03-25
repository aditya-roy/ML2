library(leaps) 
library(randomForest)

dat = read.csv(file="C:\\Users\\adity\\Downloads\\Q3_dataset.csv") 
X = dat 
X$Y <- NULL 
Y = dat$Y


K=10
set.seed(1) 
N = nrow(X) 
P = ncol(X) 
folds = cut(1:N, K, labels=FALSE) 
bics = matrix(NA, nrow=K, ncol=10) 
vars = matrix(NA, nrow=K, ncol= P)   # fill in the blank for ncol 
for(k in 1:K){ 
  itrain = which(folds!=k)
  fwd.mod = regsubsets(x=X[itrain,], y=Y[itrain], method = "forward", nvmax = 10)
  bics[k,] = summary(fwd.mod)$bic #Stores the BIC values returned by regsubsets() for each CV fold
  vars[k,] = summary(fwd.mod)$which[which.min(bics[k,]),-1] # Stores the variables used in the BIC-optimal model
}
apply(bics, 1, which.min)#3 is the most frequent model
100*apply(vars, 2, mean)#pretty clear selection
#X1, X2, X5, X8 are found to be important. The first 3 (X1, X2, X5) are selected everytime and
#X8 in 30% of the CV loops.


rf.out = randomForest(Y~., data=X)
rf.out$mtry
rf.out$importance
nms = colnames(X)
nms[order(rf.out$importance, decreasing = TRUE)]

##Answer g
'''
->Stepwise Selection is more adequate if the relationship betwen
X and Y is purely linear, and output would be highly repliable(N>P)
-> RF and linear regression models would capture different aspects
of the data if the X-Y relationship was not linear; this would become 
apparent in the selection of "less important" predictore contributions
in the models
-> RF was not cross-validated, and hence the output variable importance
assessment is not adjusted for the variablitiy in the data. by applying
CV we could end up with an assessment of variable importance that would
be comparable to that obtained from step-wise selection
-> Choice of criterion: Stepwise selection was made on the basis of AIC,
RF on the basis of decrease in MSE - so different criteria are applied to "rate"
the covariates
'''


