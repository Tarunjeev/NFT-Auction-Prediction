# NFT-Auction-Prediction
1 INTRODUCTION


This module studies the auction prices of NFT’s using various machine learning methodologies. Data about the NFT’s is obtained from the following Kaggle competition. The goal of this module is to use the historical selling records of various NFT’s and predict the price for unseen NFT’s.
Before conducting any prediction modeling, pre-processing of the data was done to handle missing-data and irregular data points. This is an important step in the project and it likely had a substantial impact on our MAE scores. After prepossessing the data, we were able to use machine learning techniques such as Multiple Linear Regression, Ada-Boost Regression, Ensemble Regression, and XG Boost Regression to choose the method with the lowest RMSE score. The method that provided our team with the lowest RMSE score, equal to 103.76218, is XG Boost.

2 DATA EXPLORATION

Firstly, we explored both of the datasets in an attempt to find patterns and peculiarities. We checked if the datatype for each column and see if any column has non-null values, after which we looked into the percentage of missing values to get a good idea about the significance of variables. Further more We had a look at the skewness of the data and the outliers present in the data.
Figure 1: info() on the data
Through the exploratory methods, we found insightful patterns in the data-set:
• Thefee1andfee2variablesresemblegasfees.
• The data had mostly slightly left skewed columns (between 0.5 and 1) and some highly right skewed columns(value > 1).
• Variables‘total’and‘X.sales’haveextremeoutliers.
This process allowed us to understand the dataset better and guided us on what to do in our next step, data- preprocessing.

 3 PREPROCESSING
 
In this project we got to explore two new kinds of data text and images, inorder to make them machine ready, we made some transformed the data to numbers. For the image data, we converted the image to its dimensions, mean of all three color layers (RGB) separately and together.
For the text data, we extracted the the top 10 theme keywords out of every row of the description by us- ing Latent Dirichlet allocation. And further expanded in that avenue by find thing the top keywords out of the description by grouping the description by the author. In order to do this I used TF-IDF [Ganesan, 2020b] and cleared the stopwords by creating a custom stopwards list [Ganesan, 2020a].
Lastly we filled the following missing value variables, ‘fee1’, ‘fee2’, ‘symbol’, ‘version’. For fee1 and fee2, I used a mode based imputer as we now that the gas prices have low variability throughout the day. For symbol and version I used a random forest based classifier to predict the values.

4 MODELS

Method
Multiple Linear Regression Lasso regression Random Forest regression LightGBM regression XG boost regression
MAE
9.512 9.564 9.417 9.342 9.294

Table 1: Table of various methods and their MAE.

4.1 Multiple Linear Regression

Firstly, based on the lectures we tried various forms of Linear regression. Starting from regressing total variable with row numbers. This served as a base line for all further approaches. Next, we began Linear regression on basis of multiple variables. We observed a decrease in score from the baseline MAE score obtained from the aforementioned method on the Kaggle Leaderboard. Within the testing of our linear regression model, we made sure that the distribution and the variance of the dependent variable was normalized and constant for all values of the independent variable.
During our exploration process, we tried to determine a mathematical relationship among several random variables which further helped us examine how multiple independent variables were related to our dependent variable ‘total’. Once all of the independent factors were determined, the information on the multiple variables was used to create prediction of the dependent variable on the level of effect that they have on the outcome variable. Notably, the information on the multiple variables can be used to create an accurate prediction based on the impact they have on the outcome variable.

 4.2 Lasso regression
 
Our next step was to use a regularization technique, because using shrinkage can help us obtain the subset of predictors that minimizes prediction error for quantitative response variable. Now, just to make sure we were using right value of alpha, we performed a grid search on couple of values of alpha, but after running the model we got worse MAE as compared to linear regression. Since, there was not much collinearity in the data using which the model could give us better results.

4.3 Random Forest regression

The third approach was Random Forest regression. Transitioning from linear regression to Random Forest allows us to secure non-linear relationships, which produces a better prediction accuracy.Since the algorithm creates a random sample of multiple decision trees and merges them together to obtain a more stable and accurate prediction through cross validation.
Additionally, on performing hyperparameter tuning using grid search, we were able to figure out a much better result of MAE equal to 9.41, since the model was not only able to handle the outliers and unexpected changes in the data points, but could also maintain the accuracy of a large proportion of data. Lastly because of more trees, the model had the power to handle a large data set with higher dimensionality.

4.4 LightGBM regression

Our next step was to use a regularization technique, because using shrinkage can help us obtain the subset of predictors that minimizes prediction error for quantitative response variable. Now, just to make sure we were using right value of alpha, we performed a grid search on couple of values of alpha, but after running the model we got worse MAE as compared to linear regression. Since, there was not much collinearity in the data using which the model could give us better results.

4.5 XG boost regression

XGBoost is a decision tree based ensemble algorithm that uses a gradient boosting framework. It utilizes decision trees, bagging and boosting, which is the process of minimizing the errors of the previous models while boosting the influence of better performing models. For this module, we trained a XGBoost model using a loss function. It outperformed all other attempted models. Consequently, we began hyperparamter tuning it on max_depth, learning_rate, n_estimators, colsample_bytree using a Grid search CV.
to boost our results we did the hyperparameter tuning on the parameters using grid search cross validation method on the parameter including colsample_bytree, max_depth, min_child_weight, eta, subsample objective and each pair of parameters was founded based on the model that give lowest MAE which overall at the end outperformed all the models and got the score for 9.456

 5 CONCLUSION
 
 5.1 Results

Nevertheless, various methods were considered and implemented to minimize the MAE score using the NFT Auctions dataset. We explored various pre-processing methodologies together as a team and our best model proved to be a XG Boost algorithm, providing us an MAE of 9.294 on Kaggle.

5.2 Lessons Learned

Through the course of this project, we learned about data prepossessing and fully comprehend it’s significance on the MAE score. We got the opportunity to work with various datasets ranging from image data to text data for the first time in a real world setting and to research more about prediction and data imputation methods unique to image and text problems. Moreover, we also learned about deep learning and quantitle method of predictions.

6 THE CHALLENGE QUESTION

The challenge question is to determine whether price of Ethereum at NFT creation date and the auction price has a relationship or not. As shown in the figure 3 below, we can observe no relationship. In efforts to tackle this skewness in data I did a log transformation on the y axis, even after this augmentation we can see no visible relationship.


We hypothesize from Figure 3 that there is an no effect of price of Ethereum at NFT creation date and the auction price and in order to further explore it we apply a t-test [R., 2021] and pearson correlation test. Note that μ0 and μ1 are means of the ethereum price and the auction price, respectively.
H0 :μ0 =μ1, Ha : μ0 ̸= μ1
Figure 3: Results of the hypothesis test
An alpha level of 0.05 is selected for the hypothesis testing. The p-value for the two tailed test is equal to 2.33424934695271e-101. This p-value is less than 0.05 and thus, we reject the null hypothesis and conclude that the ethereum price and the auction price have no relation.

 THE BONUS QUESTION
 
The bonus question is to determine whether which one of the thumbnail is created by a kid and is not an NFT. In order to do this, I have used a Deep neural network architecture. For the model to run , we need both a training and testing set. Since, we are trying to find an image, I refrained from including gif files in both training and testing sets.
For the training set, I created a dataframe by combining 500 real NFT’s [Haan, 2021] and 500 normal drawing images [Danil, 2018] with labels 1 and 0 respectively.
For the testing test, I used all the images provided to use excluding files with g ̇if extension.
(a) Results of the DNN (b) Images from the created train set Figure 4: Histogram of RNA counts Vs. Case counts
After running the model for multiple iterations, our model gave the aforementioned ids with their probabilities of not being a NFT and being an NFT respectively. From this we chose the id which has the highest probability of not being an NFT.

 REFERENCES
 
[Danil,2018] Danil(2018).Artimages:Drawing/painting/sculptures/engravings.https://www.kaggle.com/ thedownhill/art-images-drawings-painting-sculpture-engraving.
[Ganesan,2020a] Ganesan,K.(2020a).Tipsforconstructingcustomstopwordlists.http://kavita-ganesan. com/tips-for-constructing-custom-stop-word-lists/#.YWoMamLMK70.
[Ganesan, 2020b] Ganesan, K. (2020b). Tutorial: Extracting keywords with and python’s scikit-learn. https: //kavita-ganesan.com/extracting-keywords-from-text-tfidf/#.YWoG12LMK70.
[Haan,2021] Haan, A. d. (2021). Nft art collection 2021. https://www.kaggle.com/vepnar/ nft-art-dataset?select=dataset.
[R.,2021] R.,L.(2021).T-test-performinghypothesistestingwithpython.https://www.analyticsvidhya. com/blog/2021/07/t-test-performing-hypothesis-testing-with-python/.
