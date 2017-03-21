**Data Mining Case Study**

**Boston and German Credit Score data**

**Boston Housing Data**

***Executive Summary***

The Boston Housing data was collected in 1978 to discover whether air quality influenced the values of houses in Boston.   
Several neighborhood variables were selected to attempt to determine which were the best to explain housing values.   
The Boston housing Data  reports the median value of owner-occupied homes in about 500 U.S. census tracts in the Boston area,  
together with several variables which might help to explain the variation in median value across tract.   
The original study hoped to focus on air quality as the explanatory variable, however analysis indicates that other   
variables may have greater influence on housing prices.

| **Variable** | **Description** |
| --- | --- |
| crim | per capita crime rate by town |
| zn | proportion of residential land zoned for lots over 25,000 sq.ft |
| indus | proportion of non-retail business acres per town |
| chas | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) |
| nox | nitric oxides concentration (parts per 10 million) |
| rm | average number of rooms per dwelling |
| age | proportion of owner-occupied units built prior to 1940 |
| dis | weighted distances to five Boston employment centres |
| rad | index of accessibility to radial highways |
| tax | full-value property-tax rate per USD 10,000 |
| ptratio | pupil-teacher ratio by town |
| black | 1000(B - 0.63) where B is the proportion of blacks by town |
| lstat | percentage of lower status of the population |
| medv | median value of owner-occupied homes in USD 1000&#39;s |

Hence forth above acronyms will be used for variable names

**Goal**

The purpose behind this activity was to compare different statistical models, suffice their requirements, optimize to get the best parameters for 75% of training sample (random seeded based on MNumber). Model Comparison based on MSE, R-squared criteria is done for both in-sample (training set) and out-sample (test sample)

**Approach**

Various models are fitted for this kind of data where the variable of interest is continuous (median values for the pricing of houses).  
Thus this is regression problem and Linear Regression (using Generalize linear models) Regression Tree(CART) Generalized additive models  
and Neural networks is fitted for analysis of their performance.

Initial Exploratory Analysis was conducted as a part of previous homework where correlation between variables like **taxt** **indus**  
turned out to be significant. However for the scope of this case, it was assumed that Exploratory analysis was already done.  

Variable Selection for Linear regression using the stepwise was conducted to get only those variables that are significant.  
Cross Validation on entire data set using k=5 was achieved to check the model.   
For Regression Tree no pruning was done as the tree seemed to small and have less nodes and branches.  
For GAMs splines are fitted on continuous variables by using general spline option.   
For Neural Networks both scaled and unscaled data is fitted just to see whether the input to data will make   
a difference in model fit ( and yes it does)

**Major Findings**

While doing regression it was observed from the residual plots that the heteroskedasticity is present.   
So in order to check if by doing some transformation I can mitigate it, hence is transformed medv to log(medv) and regressed,   
but the presence of heteroskedasticity was just suppressed but not removed completely,   
either a different transformation is required for this and is beyond the scope for this case

**Results**

| **Method** | **In-sample MSE** | **Out-Sample MSE** |
| --- | --- | --- |
| GLM | 19.07758 | 30.73509 |
| Regression Tree | 11.40524 | 37.27731 |
| GAM | 6.20655 | 26.59801 |
| Artificial Neural Networks- Unscaled | 59.42404 | 94.9931 |
| Artificial Neural Networks- scaled | 3.467056 | 28.83375 |

As you can see from the above table the out-sample performance for each of the model is similar.  
However as noted before the presence of heteroskedasticity violates the assumption for GLM, which may also   
have affected the regression trees. GAM proves to be best in this case, giving the least TRAIN and TEST MSE.

Scaling affects the performance of ANNs and it is proved above by the huge difference in MSE.  
For this particular problem, we should proceed with GAM

**Generalized Linear Regression**

the  **generalized linear model**  ( **GLM** ) is a flexible generalization of ordinary  [linear regression](https://en.wikipedia.org/wiki/Linear_regression) that allows for response variables that have error distribution models other than a  [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution). The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a _link function_ and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.

In our case the dependent variable is still considered linear in relationship with the independent variables

Stepwise variable selection method was used with AIC as the selection criteria between null model(no variables) and full model (all the variables). The optimal model was obtained with AIC of 2214.9 with variables. It is done Training set

lstat + rm + ptratio + chas + dis + nox + crim + zn + black + rad + tax

**Cross validation:**

This is considering the entire data set and assessing the model if it the model is biased towards the test set. A 5 fold cross validation is done using cv.glm (K=5 usual standards)

An prediction error of about 22.79 that is adjusted for LOOCV bias is obtained meaning the model does not show any anomaly

**Results**

TrainMSEGLM: 19.07758

TestMSEGLM: 30.73509



**Regression Trees**

All regression techniques contain a single output (response) variable and one or more input (predictor) variables. The output variable is numerical. The general regression tree building methodology allows input variables to be a mixture of continuous and categorical variables. A decision tree is generated when each decision node in the tree contains a test on some input variable&#39;s value. The terminal nodes of the tree contain the predicted output variable values.

As you can see the split wrt to rm, lstat , rad, nox are used at nodes at different levels. While the node at the end of the tree will be mean value for that set, which will also be predicted if the observation happens to belong in that node. This is one of the disadvantage of regression tree as we will always get the mean of that node and not the actual value. However trees can be articulated precisely

**Results**

Train MSE Tree: 11.40524, Test MSE Tree: 37.27731

**Generalized Additive Model**

**generalized additive model (GAM)** is a  [generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model) in which the linear predictor depends linearly on unknown  [smooth functions](https://en.wikipedia.org/wiki/Smooth_function) of some predictor variables, and interest focuses on inference about these smooth functions. GAMs were originally developed by  [Trevor Hastie](https://en.wikipedia.org/wiki/Trevor_Hastie) and  [Robert Tibshirani](https://en.wikipedia.org/wiki/Robert_Tibshirani) to blend properties of  [generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model) with  [additive models](https://en.wikipedia.org/wiki/Additive_model).

The main idea here Is to fit splines for continuous variables as they exhibit non linearity. Therefore splines are used for all the variables apart from those are binary or factors
**TestMSEGAM** : 26.59801

**TrainMSEGAM** : 6.20655

**Neural Networks**

Neural networks or  [connectionist](https://en.wikipedia.org/wiki/Connectionism) systems are a  [computational model](https://en.wikipedia.org/wiki/Computational_model) used in computer science and other research disciplines, which is based on a large collection of simple neural units ( [artificial neurons](https://en.wikipedia.org/wiki/Artificial_neuron)), loosely analogous to the observed behavior of a  [biological brain](https://en.wikipedia.org/wiki/Brain)&#39;s axons. Each neural unit is connected with many others, and links can enhance or inhibit the activation state of adjoining neural units. Each individual neural unit computes using summation function. There may be a threshold function or limiting function on each connection and on the unit itself, such that the signal must surpass the limit before propagating to other neurons. These systems are self-learning and trained, rather than explicitly programmed, and excel in areas where the solution or  [feature detection](https://en.wikipedia.org/wiki/Feature_detection_(computer_vision)) is difficult to express in a traditional computer program.

Here two approach is adopted; One with scaling the data and one without. ANNs are sensitive to their input hence to validate that this approach is used

 Also the optimal size of the number of hidden neuron is obtained by looping it across various values   from 1-50 and the one that has given minimum MSE is finally used.

TrainMSENNet\_unscaled: 59.42404

TrainMSENNet\_scaled: 3.467056

TestMSENNet\_unscaled: 94.9931

TestMSENNet\_scaled: 28.83375



**German Credit Scoring**

The German credit data set contains 1000 observations and 21 variables with &quot;response&quot; being the response variable.  There are no empty observations.  This data defines 20 variables that are used in credit scoring decisions for those who apply to a bank for a loan.  The bank wants to minimize risk and maximize its profit by loaning to those who are good credit risks and not loaning to those who are bad risks, however there is a cost to each wrong decision.  Cost is higher for a loan that defaults, than not loaning to someone who would pay back the loan

**Goal**

The purpose behind this activity was to compare different statistical models, suffice their requirements, optimize to get the best parameters for 75% of training sample (random seeded based on MNumber). Model Comparison based on Misclassification rate, AUC (area under the curve for ROC),Credit Cost criteria is used for both in-sample (training set) and out-sample (test sample)

**Approach**

Various models are fitted for this kind of data where the variable of interest is continuous (response variable whether default or not). Thus this is classification problem and Logistic Regression (using Generalize linear models) Classification Tree(CART) Generalized additive models and Linear Discriminant Analysis model is fitted for analysis of their performance.

Variable Selection for Generalized Linear regression using the stepwise was conducted to get only those variables that are significant. Cross Validation on entire data set using k=5 was achieved to check the model. For Classification Tree no pruning was done as the tree seemed to small and have less nodes and branches. For GAMs splines are fitted on continuous variables by using general spline option. For  LDA all available variables were used

Pcut used here is 0.1667 (1/6 as the cost is ratio 1:5)

Credit Cost is calculated based on penalizing the misclassification based on the cost 1:5 function can seen here
**Major Findings**

Results

| Method | In-sample | Out-Sample |
| --- | --- | --- |
|   | AUC | Credit Cost | Misclassification Rate | AUC | Credit Cost | Misclassification Rate |
| Logistic (GLM) | 0.84 | 0.414 | 0.3133 | 0.746 | 0.632 | 0.36 |
| Classification Tree | 0.80 | 0.02333333 | 0.5933 | 0.72 | 0..01 | 0.544 |
| GAM | 0.85 | 0.70 | 0.1946 | 0.75 | 0.98 | 0.26 |
| LDA | 0.85 | 0.429 | 0.312 | 0.749 | 0.64 | 0.352 |

As you can see from the above table the out-sample performance for each is different.

GAM is providing the least misclassification rate but the cost is high which indicates that the cases that are predicted wrong have higher penalty

So considering the cost Classification Tree is best for this data

**Generalized Linear Model (Logistic Regression)**

 logistic regression, or logit regression, or logit model [
# [1]
](https://en.wikipedia.org/wiki/Logistic_regression#cite_note-Freedman09-1) is a  [regression](https://en.wikipedia.org/wiki/Regression_analysis) model where the  [dependent variable (DV)](https://en.wikipedia.org/wiki/Dependent_and_independent_variables) is  [categorical](https://en.wikipedia.org/wiki/Categorical_variable). This article covers the case of a  [binary dependent variable](https://en.wikipedia.org/wiki/Binary_variable)—that is, where it can take only two values, &quot;0&quot; and &quot;1&quot;, which represent outcomes such as pass/fail, win/lose, alive/dead or healthy/sick. Cases where the dependent variable has more than two outcome categories may be analysed in  [multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression), or, if the multiple categories are  [ordered](https://en.wikipedia.org/wiki/Level_of_measurement#Ordinal_type), in  [ordinal logistic regression](https://en.wikipedia.org/wiki/Ordinal_logistic_regression). [
# [2]
](https://en.wikipedia.org/wiki/Logistic_regression#cite_note-wal67est-2) In the terminology of  [economics](https://en.wikipedia.org/wiki/Economics), logistic regression is an example of a  [qualitative response/discrete choice model](https://en.wikipedia.org/wiki/Discrete_choice).

Using Stepwise variables are selected for this GLM model, with AIC of 719

**Results**

TestMisGLM: 0.344 TestMisGLM.step: 0.36

TrainMisGLM: 0.312 TrainMisGLM.step: 0.3133333

creditcost.GLM.out: 0.632 creditcost.GLM.in: 0.4146667

AUC Train: 0.8483076 AUC Test: .7490926

**Cross Validation**

This is considering the entire data set and assessing the model if it the model is biased towards the test set. A 5 fold cross validation is done using cv.glm (K=5 usual standards)

An prediction error of about 0.3389333 that is adjusted for LOOCV bias is obtained meaning the model does not show any anomaly



**Classification Tree**

Classification tree methods (i.e., decision tree methods) are recommended when the data mining task contains classifications or predictions of outcomes, and the goal is to generate rules that can be easily explained and translated into SQL or a natural query language.

A Classification tree labels, records, and assigns variables to discrete classes. A Classification tree can also provide a measure of confidence that the classification is correct.

Here the class of the dependent variable is decided based on the mode of that node.

**Results**

TestMisTree: 0.544        TrainMisTree: 0.5093333

creditcost.Tree.out: 0.01  creditcost.Tree.in: 0.02333333

AUC Train: 0.80       AUC Test: .7240926

**Generalized Additive Model**

Splines are fitted for continuous variable like age, amount and duration

Results
TestMisGAM: 0.26        TrainMisGAM: 0.1946

creditcost.GAM.out: 0.98  creditcost.GAM.in: 0.70

AUC Train: 0.85       AUC Test: .75



**Linear Discriminant Analysis**

Linear discriminant analysis (LDA) is a generalization of Fisher&#39;s linear discriminant, a method used in  [statistics](https://en.wikipedia.org/wiki/Statistics),  [pattern recognition](https://en.wikipedia.org/wiki/Pattern_recognition) and  [machine learning](https://en.wikipedia.org/wiki/Machine_learning) to find a  [linear combination](https://en.wikipedia.org/wiki/Linear_combination) of  [features](https://en.wikipedia.org/wiki/Features_(pattern_recognition)) that characterizes or separates two or more classes of objects or events. The resulting combination may be used as a  [linear classifier](https://en.wikipedia.org/wiki/Linear_classifier), or, more commonly, for  [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) before later  [classification](https://en.wikipedia.org/wiki/Statistical_classification).

**Results**

TestMisLDA: 0.352        TrainMisLDA: 0.312

creditcost.LDA.out: 0.64  creditcost.LDA.in: 0.429

AUC Train: 0.85       AUC Test: .749

References:

All Definitions adopted from Wikipedia

[http://www.solver.com/regression-trees](http://www.solver.com/regression-trees)

[http://www.solver.com/classification-tree](http://www.solver.com/classification-tree)
