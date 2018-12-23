# Basics
rm(list=ls())
df[1, ]
df[c(1, 2),]
dim(df)
names(df)
attach(df)
as.factor(data)
is.na(data)
pairs(df)
summary(df)
hist(data)

# train test split
trainid <- sample(1:nrow(df), nrow(df)*0.7 , replace=F)
train <- df[trainid,]
test <- df[-trainid,]

# Linear Regression
lm.fit=lm(y~x, train)
predict(lm.fit, data, interval="confidence")
predict(lm.fit, data, interval="prediction")
plot(x, y)
abline(lm.fit, lwd=3, col="red")
par(mfrow=c(2,2))
plot(lm.fit)
library(car)
vif(lm.fit)
I(X^2)
poly(data, n)

# Logistic Regression
glm.fit=glm(y~x, train, family=binomial)
glm.probs=predict(glm.fit, test, type="response")
glm.pred=ifelse(glm.probs > 0.5, 1, 0)
table(glm.pred, test.y)
mean(glm.pred==test.y)

# Linear Discriminant Analysis
library(MASS)
lda.fit=lda(y~x, train)
lda.pred=predict(lda.fit, test)
lda.class=lda.pred$class
table(lda.class, test.y)
mean(lda.class==test.y)

# Quadratic Discriminant Analysis
qda.fit=qda(y~x, train)
qda.pred=predict(qda.fit, test)
qda.class=qda.pred$class
table(qda.class, test.y)
mean(qda.class==test.y)

# K-Nearest Neighbors
library(class)
knn.pred=knn(train.X, test.X, train.y, k=50)
#table(knn.pred, test.y)
mean(knn.pred==test.y)

# Cross-Validation
library(boot)
cv.error.10=rep(0,10)
for (i in 1:10){
  glm.fit=glm(y~poly(X, i), data)
  cv.error.10[i]=cv.glm(data, glm.fit, K=10)$delta[1]
}

# Bootstrap
boot(data, statistic, R=1000)

# Best Subset Selection
library(leaps)
regfit.full=regsubsets(y~., data, nvmax)
reg.summary=summary(regfit.full)
plot(reg.summary$bic, xlab="Number of Variables", ylab="BIC", type='l')
n = which.min(reg.summary$bic)
points(n, reg.summary$bic[n], col="red", cex=2, pch=20)

# Forward and Backward Stepwise Selection
regfit.fwd=regsubsets(y~., data, nvmax, method="forward")
summary(regfit.fwd)
regfit.bwd=regsubsets(y~., data, nvmax, method="backward")
summary(regfit.bwd)

# Ridge Regression
library(glmnet)
ridge.mod=glmnet(x[train,], y[train], alpha=0, lambda=grid, thresh=1e-12)
plot(ridge.mod)
cv.out=cv.glmnet(x[train,], y[train], alpha=0)
plot(cv.out)
bestlam=cv.out$lambda.min
ridge.pred=predict(ridge.mod, s=bestlam, newx=x[test,])
mean((ridge.pred-y.test)^2)
out=glmnet(x, y, alpha=0)
predict(out, type="coefficients", s=bestlam)[1:20,]

# Lasso
lasso.mod=glmnet(x[train,], y[train], alpha=1, lambda=grid)
plot(lasso.mod)
cv.out=cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.out)
bestlam=cv.out$lambda.min
lasso.pred=predict(lasso.mod, s=bestlam, newx=x[test,])
mean((lasso.pred-y.test)^2)
out=glmnet(x, y, alpha=1, lambda=grid)
lasso.coef=predict(out, type="coefficients", s=bestlam)[1:20,]
lasso.coef[lasso.coef!=0]

# Principal Components Regression
library(pls)
pcr.fit=pcr(y~., data, scale=TRUE, validation ="CV")
summary(pcr.fit)
validationplot(pcr.fit, val.type="MSEP")
pcr.fit=pcr(y~., data, subset=train, scale=TRUE, validation ="CV")
pcr.pred=predict(pcr.fit, x[test,], ncomp)
mean((pcr.pred-y.test)^2)

# Partial Least Squares
pls.fit=plsr(y~., data, subset=train, scale=TRUE, validation ="CV")
summary(pls.fit)
validationplot(pls.fit, val.type="MSEP")
pls.pred=predict(pls.fit, x[test,], ncomp=2)
mean((pls.pred-y.test)^2)

# Splines
library(splines)
bs(x, knots)
ns(x)
fit=smooth.spline(x, y, cv=TRUE)
fit=loess(y~x, span, data)

# GAM
library(gam)
s() # smoothing spline
lo() # local regression
gam()
plot.gam()

# Tree
library(tree)
tree.data = tree(y~., data, subset=train)
summary(tree.data)
plot(tree.data)
text(tree.data, pretty = 0)

cv.data = cv.tree(tree.data)
plot(cv.data$size, cv.data$dev, type='b')

prune.data = prune.tree(tree.data, best)
plot(prune.data)
text(prune.data, pretty = 0)

pred.data = predict(tree.data, newdata=data[-train,])
data.test=data[-train, y]
mean((data.test - pred.data)^2)

# Bagging and Random Forests
library(randomForest)
bag.data=randomForest(y~., data, subset=train, mtry, ntree, importance=TRUE)
pred.bag = predict(bag.data, newdata=data[-train,])
mean((pred.bag-data.test)^2)

importance(bag.data)
varImpPlot(bag.data)

# Boosting
library(gbm)
boost.data=gbm(y~., data[train,], distribution="gaussian")
summary(boost.data)

# SVM
library(e1071)
svmfit=svm(y~., data, kernel, cost)
plot(svmfit, data)
summary(svmfit)
tune.out=tune(svm, y~., data, kernel,
  ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))

# ROC
library(ROCR)
rocplot=function(pred, truth, ...)
{predob = prediction (pred, truth)
  perf = performance (predob , "tpr", "fpr")
  plot(perf ,...)}
svmfit.opt=svm(y~., data=dat[train,], kernel="radial", gamma=2, cost=1, decision.values=T)
fitted=attributes(predict(svmfit.opt,dat[train,], decision.values=TRUE))$decision.values
rocplot(fitted, dat[train, "y"], main="Training Data")