# Predicting-a-pulsar-star
1. The project is simply used to predict weather a star is a pulsar star or not.
2. Currently used models in the project - a) Logistic Regression b) Decision Tree
3. More models will be added with time.
4. Data set was downloaded from kaggle. Link - https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star
5. Any suggested models or techiniques would be appreciated.



# Analysis of the models 
1. The Logistic Regression model performed well to the trainind set that I chose but it does not even generalize well to the training set.
   As we were trying to fit a line and got poor results, we can safely assume that the data is not linearly separable.

2. The Decision Tree model works better than Logistic Regression model. The accuracy is significantly higher in all scenarios, but as the
   data is skewed, accuracy does not serve as a viable evaluation mettric for performance, so f1 score is used instead.It is definitely 
   better than logistic regression model but still the performance is not upto the mark.