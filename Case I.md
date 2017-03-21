**Data Mining Case Study for Boston and German Credit Score data**

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
  
  
      
  
