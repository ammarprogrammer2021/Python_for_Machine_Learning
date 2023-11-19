# Python_for_Machine_Learning
Basic and advanced implementations of Machine Learning Algorithms in Python.

## Data Analysis 

In the folder Data Analysis and Visualization. the summary of the basic knowledge that I learned  about the important python libraries such as pandas and seaborn for data analysis were presented. 

In the Capstone Project using the basic knowledge we acquired about data analysis we tried to answer the question whether there is a conflict of interest of a website that both sells movie tickets and displays movie ratings.
it's been suspected that the website named Fandango overrates its movies in order to sell more movie tickects. For this reason we used data analysis to confirm or deny this suspicion.

We compared the true ratings that Fandango holds in the backend and those that it actually displays the users to check for any discrepancies.also we compared with the ratings of other websites such as rotten tomatoes.

In the end we reached the conclusion that the Fandango ratings are obviously too high even for movies that were poorly rated by other rating websites like TAKEN 3.

## Machine Learning Concepts
we started with linear Regression considering the fact that it is one of the oldest algorithms developed in this domain.In that section of the course we covered the following topics:
* Theory of Linear Regression (it was not uploaded since it is only about mathematical formulas of linear Regression).
* Simple Implementation with Python.
* Linear Regression with Scikit-learn.
* Polynomial Regression.
* Regularization.
  
* The following subject was feature engineering and data preparation in which we learnt some simple and useful techniques to deal with some issues in data preparation such as Outliers , missing and categorical 
  data. In order to get a better idea how this is done in real world applications. we had a data set called Ames_Housing_Data. In this Data Frame the target label is SalePrice and there are many other features 
  like Living Area, Overall Quality, Overall Condition ect. This Data Set has the mentioned issues outliers, missing data , categorical data.
  In the folder missing data we walked through the some methods (Filling, Dropping Data based on rows, encoding options ect ) to alleviate thoses issues and get the data ready for a machine learning model to 
  train and predict the target label.

* after data preparation (data cleaning , splitting in train and test data, evatually scaling) we trained a linear regression model namely ElasticNet and did a GridSearch to choose the best values of the model's 
 parameters (in our case alpha and l1_ratio) to get the best possible performance.

* the other files contain the implementations in Python of the other common supervised learning algorithms such as logistic regression, K Nearest Neighbors, Support Vector Machines, Tree Based Methods (Decision 
 Tree Learning, Random Forests) and Boosting Methods like AdaBoost, Gradient Boosting with their relevant Coding Projects.

* In One of those Projects Tree based methods Capstone project we analysed the churn of an internet and telephone provider (churn means that the customer will at some point quit using the service of internet and 
  telephone and of course not paying for it). It started with typical Exploratory Data Analysis such as checking whether there is missing data , displaying the balance of target label (Churn) in which we found 
  out the label Churn is quite imbalanced the number of No Churn almost three times that of Yes Churn. we created barplot showing the correlation of some features with class label to figure out the influence of 
  those features on the target label.then we analysed the feature that is mostly correlated with Churn which happened to be tenure column (tenure means the number of months an individual was a customer by this 
  service provider) with histogram and its relationship with target label.In the end we were able to draw some useful insights like in the long term contracts the customers that are have higher charges are most 
  likely to churn from the service.
  finally we trained some tree models to predict based on the features which customer will churn. we used Decision tree, Random Forest and AdaBoost then we compared their performance judging by the metrics 
  (precision, F1-score, recall, accuracy).AdaBoost performed slightly better than the Tree models.
