
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

**Results**

| Method              | In-sample | Out-Sample  |                        |       |             |                        |
|---------------------|-----------|-------------|------------------------|-------|-------------|------------------------|
|                     | AUC       | Credit Cost | Misclassification Rate | AUC   | Credit Cost | Misclassification Rate |
| Logistic (GLM)      | 0.84      | 0.414       | 0.3133                 | 0.746 | 0.632       | 0.36                   |
| Classification Tree | 0.80      | 0.02333333  | 0.5933                 | 0.72  | 0..01       | 0.544                  |
| GAM                 | 0.85      | 0.70        | 0.1946                 | 0.75  | 0.98        | 0.26                   |
| LDA                 | 0.85      | 0.429       | 0.312                  | 0.749 | 0.64        | 0.352                  |

As you can see from the above table the out-sample performance for each is different.

GAM is providing the least misclassification rate but the cost is high which indicates that the cases that are predicted wrong have higher penalty

So considering the cost Classification Tree is best for this data

**Generalized Linear Model (Logistic Regression)**

 logistic regression, or logit regression, or [logit model](https://en.wikipedia.org/wiki/Logistic_regression#cite_note-Freedman09-1) is a  [regression](https://en.wikipedia.org/wiki/Regression_analysis) model where the  [dependent variable (DV)](https://en.wikipedia.org/wiki/Dependent_and_independent_variables) is  [categorical](https://en.wikipedia.org/wiki/Categorical_variable). This article covers the case of a  [binary dependent variable](https://en.wikipedia.org/wiki/Binary_variable)—that is, where it can take only two values, &quot;0&quot; and &quot;1&quot;, which represent outcomes such as pass/fail, win/lose, alive/dead or healthy/sick. Cases where the dependent variable has more than two outcome categories may be analysed in  [multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression), or, if the multiple categories are  [ordered](https://en.wikipedia.org/wiki/Level_of_measurement#Ordinal_type), in  [ordinal logistic regression](https://en.wikipedia.org/wiki/Ordinal_logistic_regression).In the terminology of  [economics](https://en.wikipedia.org/wiki/Economics), logistic regression is an example of a  [qualitative response/discrete choice model](https://en.wikipedia.org/wiki/Discrete_choice).

Using Stepwise variables are selected for this GLM model, with AIC of 719

**Results**

TestMisGLM: 0.344 TestMisGLM.step: 0.36

TrainMisGLM: 0.312 TrainMisGLM.step: 0.3133333

creditcost.GLM.out: 0.632 creditcost.GLM.in: 0.4146667

AUC Train: 0.8483076 AUC Test: .7490926
    
    
<hr/>     

**Cross Validation**

This is considering the entire data set and assessing the model if it the model is biased towards the test set. A 5 fold cross validation is done using cv.glm (K=5 usual standards)

An prediction error of about 0.3389333 that is adjusted for LOOCV bias is obtained meaning the model does not show any anomaly


  
<hr/>  

**Classification Tree**

Classification tree methods (i.e., decision tree methods) are recommended when the data mining task contains classifications or predictions of outcomes, and the goal is to generate rules that can be easily explained and translated into SQL or a natural query language.

A Classification tree labels, records, and assigns variables to discrete classes. A Classification tree can also provide a measure of confidence that the classification is correct.

Here the class of the dependent variable is decided based on the mode of that node.

**Results**

TestMisTree: 0.544        TrainMisTree: 0.5093333

creditcost.Tree.out: 0.01  creditcost.Tree.in: 0.02333333

AUC Train: 0.80       AUC Test: .7240926

  
  <hr/>  
  
**Generalized Additive Model**

Splines are fitted for continuous variable like age, amount and duration

Results
TestMisGAM: 0.26        TrainMisGAM: 0.1946

creditcost.GAM.out: 0.98  creditcost.GAM.in: 0.70

AUC Train: 0.85       AUC Test: .75


<hr/>  

**Linear Discriminant Analysis**

Linear discriminant analysis (LDA) is a generalization of Fisher&#39;s linear discriminant, a method used in  [statistics](https://en.wikipedia.org/wiki/Statistics),  [pattern recognition](https://en.wikipedia.org/wiki/Pattern_recognition) and  [machine learning](https://en.wikipedia.org/wiki/Machine_learning) to find a  [linear combination](https://en.wikipedia.org/wiki/Linear_combination) of  [features](https://en.wikipedia.org/wiki/Features_(pattern_recognition)) that characterizes or separates two or more classes of objects or events. The resulting combination may be used as a  [linear classifier](https://en.wikipedia.org/wiki/Linear_classifier), or, more commonly, for  [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) before later  [classification](https://en.wikipedia.org/wiki/Statistical_classification).
   
  
  
**Results**

TestMisLDA: 0.352        TrainMisLDA: 0.312

creditcost.LDA.out: 0.64  creditcost.LDA.in: 0.429

AUC Train: 0.85       AUC Test: .749
    
    
    
  
***References:***

All Definitions adopted from Wikipedia

[http://www.solver.com/regression-trees](http://www.solver.com/regression-trees)

[http://www.solver.com/classification-tree](http://www.solver.com/classification-tree)
