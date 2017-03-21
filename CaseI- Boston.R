library(MASS)
library(ggplot2)
library(GGally)
library(lmtest)
library(car)
library(rpart)
library(mgcv)
library(nlme)
library(nnet)
library(neuralnet)
data(Boston); #this data is in MASS package
colnames(Boston) 

str(Boston)
summary(Boston)
##First and foremost thing, Test and Training data
set.seed(10786158)

subset = sample(nrow(Boston), nrow(Boston) * 0.75)
Boston_train = Boston[subset, ]
Boston_test = Boston[-subset, ]

##You can even scale the data
Boston_train_scaled = Boston_train
Boston_train_scaled[c(1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13)] = scale(Boston_train_scaled[c(1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13)])
Boston_test_scaled = Boston_test
Boston_test_scaled[c(1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13)] = scale(Boston_test_scaled[c(1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13)])


##Plot Can see That indus is correlated with tax; dis and nox; ie. corr>0.7
##nox and age are correlated; nox and dis
pairs <- ggpairs(Boston_train, 
                 columns = c(1:14),
                 lower=list(continuous=wrap("smooth",
                                            colour="turquoise4")),
                 diag=list(continuous=wrap("barDiag",
                                           fill="turquoise4")))  + 
  theme(panel.background = element_rect(fill = "gray98"),
        axis.line.y = element_line(colour="gray"),
        axis.line.x = element_line(colour="gray"))
pairs

##Fit the model, Linear regression

fit1 <- lm(medv ~ ., data = Boston_train)
summary(fit1) ##indus and age are the least significant adj-r-sqd 0.7419
bptest(fit1) ##BP = 65.122, df = 13, p-value = 6.265e-09 , reject the null hypothesis that the the residuals are homoskedastic
vif(fit1) ##vif for rad is about 7.14 and tax about 9.787

##removing the variables indus, age and rad

fit2 <- lm(medv ~ .-indus-age-rad, data = Boston_train)
summary(fit2) ## adj r-squared 0.7276, tax is insignificant

fit3 <- lm(medv ~ .-indus-age-rad-tax, data = Boston_train) ## all are significant adj-r 0.7283

fity2 <- 

fitglm <- glm(medv ~ .-indus-age-rad-tax, data = Boston_train)
summary(fitglm)

## due to heteroskedasticity I am also plotting log of the medv to check ; it is better that no transformation but for consistency purpose I am using not using the transformed version
fitglmlog <- glm(log(medv)~.-indus-age-rad-tax, data = Boston_train)
plot(fitglmlog)
##insample
TrainMSEGLM = mean((Boston_train$medv - predict(fitglm))^2) ##20.22483
##out-sample
FittedGLM = predict(fitglm, newdata = Boston_test, type = "response")
TestMSEGLM =mean((Boston_test$medv - Fitted)^2) ##31.43583

###########################################################################

nullmodel = glm(medv ~ 1, data = Boston_train)
fullmodel = glm(medv ~ ., data = Boston_train)
model.step = step(nullmodel, scope = list(lower = nullmodel, upper = fullmodel), 
                  direction = "both")
boston_model_lm = glm(medv ~ lstat + rm + ptratio + chas + dis + nox + crim + zn + 
                        black + rad + tax, data = Boston_train)


boston_model_lm_alldata = glm(medv ~ lstat + rm + ptratio + chas + dis + nox + crim + zn + 
                                black + rad + tax, data = Boston)
cv.glm(data = Boston, glmfit = boston_model_lm_alldata, K = 5)$delta[2]##23.79

TrainMSEGLM = mean((Boston_train$medv - predict(boston_model_lm))^2) ##20.22483
##out-sample
FittedGLM = predict(boston_model_lm, newdata = Boston_test, type = "response")
TestMSEGLM =mean((Boston_test$medv - FittedGLM)^2) ##31.43583

########################################################################################
#Regression Tree

fitTree <- rpart(formula = medv ~ ., data = Boston_train)
summary(fitTree)
sn

##insample
TrainMSETree = mean((Boston_train$medv - predict(fitTree))^2) ##11.40524

##out-sample
FittedTree = predict(fitTree, newdata = Boston_test)
TestMSETree =mean((Boston_test$medv - FittedTree)^2) ##37.277

#########################################################################################
## Generalized additive Model
gam_formula <- as.formula(paste("medv~s(crim)+s(zn)+s(indus)+s(nox)+s(rm)+s(age)+s(dis)+s(tax)+s(ptratio)+s(black)+s(lstat)+", paste(colnames(Boston_train)[c(4,9)], collapse = "+")))

fitGAM <- gam(formula = gam_formula, data = Boston_train)
summary(fitGAM) ## R-adj 75.1 with age and indus insignificant

TrainMSEGAM = mean((Boston_train$medv - predict(fitGAM))^2) ##11.40524

##out-sample
FittedGAM = predict(fitGAM, newdata = Boston_test)
TestMSEGAM =mean((Boston_test$medv - FittedGAM)^2) ##31.190


par(mfrow=c(2,2))
plot(fitGAM, shade = TRUE,  seWithMean = TRUE, scale = 0)




#########################################################################


##NN
###Unscaled
check <- data.frame(n = integer(), mse = double())
for (i in 1:50) {
  fit_NN <- nnet(medv ~ ., size = i, data = Boston_train, linout = TRUE)
  predict_nnet = predict(fit_NN, Boston_test)
  check[i,1] = i
  check[i,2] = mean((predict_nnet - Boston_test$medv)^2)
}
which.min(check$mse)

fit_NN_unscaled <- nnet(medv ~ ., size = 14, data = Boston_train, linout = TRUE)

###scaled
check1 <- data.frame(n = integer(), mse = double())
for (i in 1:50) {
  fit_NN <- nnet(medv ~ ., size = i, data = Boston_train_scaled, linout = TRUE)
  predict_nnet = predict(fit_NN, Boston_test_scaled)
  check1[i,1] = i
  check1[i,2] = mean((predict_nnet - Boston_test_scaled$medv)^2)
}
which.min(check1$mse)
fit_NN_scaled <- nnet(medv ~ ., size = 28, data = Boston_train_scaled, linout = TRUE)



##Neural Networks
fittedNNet <- nnet(medv~. ,data=Boston_train,size=1, maxit = 100 )


fittedNNet <- neuralnet(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, Boston_train, hidden = 1,rep = 150)
wts.in<-fittedNNet$wts
struct<-fittedNNet$n
plot.nnet(wts.in,struct=struct)
plot.nnet(fittedNNet)


##insample 
TrainMSENNet <-mean(Boston_train$medv  - predict(fittedNNet)) ##21.1762533
TestMSENNet <- mean(Boston_test$medv - predict(fittedNNet, Boston_test)) ##22.59685039

predict_nn_in_unscaled = predict(fit_NN_unscaled, Boston_train)
TrainMSENNet_unscaled <- mean((predict_nn_in_unscaled - Boston_train$medv)^2)

predict_nn_in_scaled = predict(fit_NN_scaled, Boston_train_scaled)
TrainMSENNet_scaled <- mean((predict_nn_in_scaled - Boston_train_scaled$medv)^2)


##outsasmple


predict_nn_out_unscaled = predict(fit_NN_unscaled, Boston_test)
TestMSENNet_unscaled <- mean((predict_nn_out_unscaled - Boston_test$medv)^2)

predict_nn_out_scaled = predict(fit_NN_scaled, Boston_test_scaled)
TestMSENNet_scaled <- mean((predict_nn_out_scaled - Boston_test_scaled$medv)^2)





