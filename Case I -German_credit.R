##Credit Scoring
library(Matrix)
library(foreach)
library(glmnet)
library(verification)
library(rpart)
library(ROCR)
library(mgcv)


german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

colnames(german_credit) = c("chk_acct", "duration", "credit_his", "purpose", 
                            "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                            "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                            "job", "n_people", "telephone", "foreign", "response")

# orginal response coding 1= good, 2 = bad we need 0 = good, 1 = bad
german_credit$response = german_credit$response - 1
set.seed(10786158)
subset <- sample(nrow(german_credit), nrow(german_credit) * 0.75)
german_credit_train = german_credit[subset, ]
german_credit_test = german_credit[-subset, ]

colnames(german_credit)

creditcost <- function(observed, predicted) {
  weight1 = 5
  weight0 = 1
  c1 = (observed == 1) & (predicted == 0)  #logical vector - true if actual 1 but predict 0
  c0 = (observed == 0) & (predicted == 1)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}

pcut=0.1667
cost1 <- function(r, pi) {
  mean(((r == 0) & (pi > pcut)) | ((r == 1) & (pi < pcut)))
}

#########################################################################################
##GLM
fittedGLM<- glm(response~ . , family = binomial, german_credit_train)
glm.step <- step(fittedGLM, k = 2, direction = c("both"))
fittedGLM.cv <- cv.glm(german_credit_train, glm.step,K=5, cost=cost1)
hist(predict(fittedGLM))
hist(predict(fittedGLM, type = "response"))
table(predict(fittedGLM, type = "response") > pcut)


##lasso for glm giving error as the data is imbalanced
#lasso_fit = glmnet(x = as.matrix(german_credit_train[, 1:20]), y = german_credit_train[, 21], family = "binomial", alpha = 1)#coef(lasso_fit, s = 0.02)

##step wise with Logistic to get the best model with AIC criteria

##glm.step <- step(fittedGLM.cv, k = log(nrow(german_credit_train)), direction = c("both"))

##insample - GLM
fittedGLM.insample.prob <- predict(fittedGLM, german_credit_train, type = "response")
fittedGLM.insample <- as.numeric(fittedGLM.insample.prob > pcut)
table(german_credit_train$response, fittedGLM.insample, dnn = c("Truth", "Predicted"))
TrainMisGLM <- mean(ifelse(german_credit_train$response != fittedGLM.insample, 1, 0))##0.50933


##insample - GLM Step 
fittedGLM.step.insample.prob <- predict(glm.step, german_credit_train, type = "response")
fittedGLM.step.insample <- as.numeric(fittedGLM.step.insample.prob > pcut)
table(german_credit_train$response, fittedGLM.step.insample, dnn = c("Truth", "Predicted"))
TrainMisGLM.step <- mean(ifelse(german_credit_train$response != fittedGLM.step.insample, 1, 0))##0.5266 ##0.512


##out-sample - GLM
fittedGLM.outsample.prob <- predict(fittedGLM, german_credit_test, type = "response")
fittedGLM.outsample <- as.numeric(fittedGLM.outsample.prob > pcut)
table(german_credit_test$response, fittedGLM.outsample, dnn = c("Truth", "Predicted"))
TestMisGLM <- mean(ifelse(german_credit_test$response != fittedGLM.outsample, 1, 0))

##outsample - GLM Step
fittedGLM.step.outsample.prob <- predict(glm.step, german_credit_test, type = "response")
fittedGLM.step.outsample <- as.numeric(fittedGLM.step.outsample.prob > pcut)
table(german_credit_test$response, fittedGLM.step.outsample, dnn = c("Truth", "Predicted"))
TestMisGLM.step <- mean(ifelse(german_credit_test$response != fittedGLM.step.outsample, 1, 0))

creditcost.GLM.out <- creditcost(german_credit_test$response, fittedGLM.outsample)

creditcost.GLM.in <- creditcost(german_credit_train$response, fittedGLM.step.insample)

##insample AUC
roc.plot(german_credit_train$response == "1", fittedGLM.step.insample.prob)
AUCGLM.in <- roc.plot(german_credit_train$response == "1", fittedGLM.step.insample.prob)$roc.vol 

##outsample AUC
roc.plot(german_credit_test$response == "1", fittedGLM.outsample.prob)
AUCGLM.out <- roc.plot(german_credit_test$response == "1", fittedGLM.outsample.prob)$roc.vol 


#######################################################################################s

###Tree

fitTree <- rpart(formula = response ~ ., data = german_credit_train, method = "class")
summary(fitTree)
plot(fitTree)
text(fitTree)

##insample

fittedTree.prob.insample = predict(fitTree, german_credit_train, type = "prob")
fittedTree.insample <- as.numeric(fittedTree.prob.insample > pcut)
TrainMisTree <- mean(ifelse(german_credit_train$response != fittedTree.insample, 1, 0)) 

##out-sample
fittedTree.prob.outsample = predict(fitTree, german_credit_test, type = "prob")
fittedTree.outsample <- as.numeric(fittedTree.prob.outsample > pcut)
TestMisTree <- mean(ifelse(german_credit_test$response != fittedTree.outsample, 1, 0))

##Creditcost
creditcost.TREE.out <- creditcost(german_credit_test$response, fittedTree.prob.outsample)

creditcost.TREE.in <- creditcost(german_credit_train$response, fittedTree.prob.insample)


###AUC
##insample
  pred = prediction(fittedTree.prob.insample[,2], german_credit_train$response )
  perf = performance(pred, "tpr", "fpr")
  plot(perf, colorize = TRUE)
  AUcTree.in <- slot(performance(pred, "auc"), "y.values")[[1]]
  
##outsample
  pred = prediction(fittedTree.prob.outsample[, 2], german_credit_test$response)
  perf = performance(pred, "tpr", "fpr")
  plot(perf, colorize = TRUE)
  AUcTree.out <- slot(performance(pred, "auc"), "y.values")[[1]]

#################################################################################

##GAM
fitGAM <- gam(response~chk_acct+s(duration)+credit_his+purpose+s(amount)+saving_acct+present_emp+installment_rate+sex+other_debtor+present_resid+property+s(age)+other_install+housing+n_credits+job+n_people+telephone+foreign, data=german_credit_train, family = "binomial")
summary(fitGAM) ## R-adj 75.1 with age and indus insignificant


## Create a formula for a model with a large number of variables:
gam_formula <- as.formula(paste("response~s(duration)+s(amount)+s(age)+chk_acct+credit_his+purpose+saving_acct+present_emp+installment_rate+sex+other_debtor+present_resid+property+other_install+housing+n_credits+job+n_people+telephone+foreign"))

fitGAM <- gam(formula = gam_formula, family = binomial, data = german_credit_train)
par(mfrow=c(2,2))
plot(fitGAM, shade = TRUE,seWithMean = TRUE, scale = 0)

summary(fitGAM)
##insample
fittedGAM.insample.prob <- predict(fitGAM)
fittedGAM.insample <- as.numeric(fittedGAM.insample.prob > pcut)
TrainMisGAM <- mean(ifelse(german_credit_train$response != fittedGAM.insample, 1, 0))

##out-sample
fittedGAM.outsample.prob <- predict(fitGAM, german_credit_test)
fittedGAM.outsample <- as.numeric(fittedGAM.outsample.prob > pcut)
TestMisGAM <- mean(ifelse(german_credit_test$response != fittedGAM.outsample, 1, 0))
par(mfrow=c(2,2))
plot(fitGAM, shade = TRUE,  seWithMean = TRUE, scale = 0)

par(mfrow=c(1,1))
##insample AUC
roc.plot(german_credit_train$response == "1", fittedGAM.insample.prob)
AUCGAM.in <- roc.plot(german_credit_train$response == "1", fittedGAM.insample.prob)$roc.vol 
##outsample AUC
roc.plot(german_credit_test$response == "1", fittedGAM.outsample.prob)
AUCGAM.out <- roc.plot(german_credit_test$response == "1", fittedGAM.outsample.prob)$roc.vol 



##Creditcost
creditcost.GAM.out <- creditcost(german_credit_test$response, fittedGAM.outsample)

creditcost.GAM.in <- creditcost(german_credit_train$response, fittedGAM.insample)



#############################################################################################
##LDA

fitLDA <- lda(response~ ., data = german_credit_train)
lda.in <- predict(fitLDA)
pcut.lda <- pcut
pred.lda.in <- (lda.in$posterior[, 2] >= pcut.lda) * 1
table(german_credit_train$response, pred.lda.in, dnn = c("Obs", "Pred"))

TrainMisLDA <- mean(ifelse(german_credit_train$response != pred.lda.in, 1, 0))


lda.out <- predict(fitLDA, newdata = german_credit_test)
cut.lda <- pcut
pred.lda.out <- as.numeric((lda.out$posterior[, 2] >= cut.lda))
table(german_credit_test$response, pred.lda.out, dnn = c("Obs", "Pred"))

TestMisLDA <- mean(ifelse(german_credit_test$response != pred.lda.out, 1, 0)) ##0.352


##insample AUC
roc.plot(german_credit_train$response == "1", lda.in$posterior[, 2])
AUCLDA.in <- roc.plot(german_credit_train$response == "1", lda.in$posterior[, 2])$roc.vol 
##outsample AUC
roc.plot(german_credit_test$response == "1", lda.out$posterior[, 2])
AUCLDA.out <- roc.plot(german_credit_test$response == "1", lda.out$posterior[, 2])$roc.vol 


##Creditcost
creditcost.LDA.out <- creditcost(german_credit_test$response, pred.lda.out)

creditcost.LDA.in <- creditcost(german_credit_train$response, pred.lda.in)
