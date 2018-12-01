# Basics
rm(list=ls())
df[1, ]
df[c(1, 2),]
dim(df)
names(df)
attach(df)
as.factor(data)
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
