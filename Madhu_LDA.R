[10:39 PM] Jaishankar, Madhu Mohan
dat = iris
dat$Species = ifelse(dat$Species=="virginica",1,0)dat$Species = as.factor(dat$Species)levels(dat$Species) = c("other","virginica")
# check for normalitypar(mfrow=c(2,4))
for(j in 1:4){
  hist(dat[which(dat$Species=="other"),j], col="pink", xlab="predictor for other", main=names(dat)[j])
  hist(dat[which(dat$Species=="virginica"),j], col="cyan", xlab="predictor for virginca", main=names(dat)[j])
}
# check for variance
# H0: variance is equalfor(j in 1:4){
print(bartlett.test(dat[,j]~dat$Species)$p.value)
}# LDA
library(MASS)
lda.o = lda(Species~., data= dat)
(lda.o)# QDA
qda.o = qda(Species~., data= dat)lda.pred = predict(lda.o, newdata = dat)# confusion matrix
(tb.lda = table(lda.pred$class,dat$Species))# accuracy
sum(diag(tb.lda))/sum(tb.lda)

