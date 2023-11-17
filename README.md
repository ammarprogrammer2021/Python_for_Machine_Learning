# Python_for_Machine_Learning
Basic and advanced implementations of Machine Learning Algorithms in Python.

## Data Analysis 

In the folder Data Analysis and Visualization. the summary of the basic knowledge that I learned  about the important python libraries such as pandas and seaborn for data analysis were presented. 

In the Capstone Project using the basic knowledge we acquired about data analysis we tried to answer the question whether there is a conflict of interest of a website that both sells movie tickets and displays movie ratings.
it's been suspected that the website named Fandango overrates its movies in order to sell more movie tickects. For this reason we used data analysis to confirm or deny this suspicion.

We compared the true ratings that Fandango holds in the backend and those that it actually displays the users to check for any discrepancies.also we compared with the ratings of other websites such as rotten tomatoes.

In the end we reached the conclusion that the Fandango ratings are obviously too high even for movies that were poorly rated by other rating websites like TAKEN 3.

## Machine Learning Concepts
### Linear Regression:
we started with linear Regression considering the fact that it is one of the oldest algorithms developed in this domain.In that section of the course we covered the following topics:
* Theory of Linear Regression (it was not uploaded since it is only about mathematical formulas of linear Regression).
* Simple Implementation with Python.
* Linear Regression with Scikit-learn.
* Polynomial Regression.
* Regularization.
  
The following subject was feature engineering and data preparation in which we learnt some simple and useful techniques to deal with some issues in data preparation such as Outliers in data, missing data and categorical data. In order to get a better idea how this is done in real world application. we had a data set called Ames_Housing_Data. In this Data Frame the target label is SalePrice and there are many other features like Living Area, Overall Quality, Overall Condition ect. This Data Set has the mentioned issues outliers, missing data , categorical data.
In the folder missing data we walked through the some methods (Filling, Dropping Data based on rows, encoding options ect ) to alleviate thoses issues and get out data set ready for a machine learning model to train and predict the target label.

after data preparation (data cleaning , splitting in train and test data, evatually scaling) we trained a linear regression model namely ElasticNet and did a GridSearch to choose the best values of the model's parameters (in our case alpha and l1_ratio) to get the best possible performance.
