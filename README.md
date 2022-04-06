# Analyzing-ML-Regression-Models-for-Price-Prediction-of-Amsterdam-North-Holland-Airbnb

This project aims to study which algorithms performs best in predicting the optimum price listing using Machine Learning (ML) algorithms by doing comparison between different regression models. The best model can then be used to predict price for the listings and help owners and customers. This project analyzes the performance of Linear regression, Decision tree and XGBoost Regressor in predicting genuine price. Few hypotheses testing have also been carried out in this project.

## Dataset:
The dataset is acquired from http://insideairbnb.com/get-thedata.html [5] which has publicly available dataset for various countries and regions. 
Data for Amsterdam, North Holland was extracted and annalyzed for the time period of Nov 2021 till Oct 2022. 
The dataset comprises of 3 csv files: 
1. Calendar .csv which contains information of 1 year (from 2021 to 2022) and includes factors like date, availability, price etc. 
2. Listing .csv which contains details about the shared property such as number of beds, rooms, property type, amenities etc.
3. Reviews.csv which contains reviews and feedbacks from the customers which would help in sentiment analysis.
The total records contained in data is around 2 million and we have combined all 3 files based on the id field. 

## Data Pre-Processing:
Each feature of the dataset was inspected to:
(i) remove features with frequent and irreparable missing fields or set the missing values to zero where appropriate
(ii)convert some features into floats (e.g., by removing the dollar sign in prices)
(iii) change Boolean features to binaries
(iv) remove irrelevant or uninformative features, e.g., host picture URL, constant valued fields,duplicate features

All preprocessed csv files were merged into one single large file. To implement machine learning models, encoding was done on categorical features of the dataset using Label Encoder from ‘Sklearn’ which is inbuilt package in python. Sentiment polarity/score for reviews of each listing was done  using ‘TextBlob’ library of python. TextBlob assigns a sentiment score between -1 (very negative sentiment) and 1 (very positive sentiment) to each analyzed comment. After initial data cleaning, data exploration was performed.

## Hypothesis Tests:
1. Tested distribution of price using Shapiro Test
2. Tested if the variance of all room types (private, shared room, hotel and entire home are equal.
3. Performed One ANOVA Test to determine if mean price is same for listings across neighborhoods
4. Performed Chi-Square Test to test relation between room type and neighborhood since both are categorical variable

## Model Implementation:
After reducing the dimensionality by dropping the column which were highly correlated, top 10 features were selected, utilizing the ‘SelectKBest’ method and f_classify function which are offered by ‘sklearn.feature_selection’ python library. SelectKBest selects features according to the K highest score and it uses f_classify to compute the scores. The ‘f_classify’ function compare the features and compute ANOVA F-value of the provided features. The function returns F-statistic for each feature and p-values associated with the F-statistic 
RMSE (Root Mean Squared Error) was used as an evaluation factor for all the 3 models. RMSE should be less than standard deviation of the target variable for a model to work better.
1. Method 1: Linear regression
2. Method 2: Random Forest Regressor
3. Method 3: XG Boost Regressor

4-fold validation technique was implemented to verify the accuracy of XG Boost Regressor model. The result shows low standard deviation (2.38) and RMSE (28.046).

## References:
[1]	Choudhary, Paridhi, Aniket Jain and Rahul Baijal. “Unravelling Airbnb Predicting Price for New Listing.” arXiv: General Finance (2018): n. pag.
[2]	Kalehbasti, Pouya Rezazadeh, Liubov Nikolenko and Hoormazd Rezaei. “Airbnb Price Prediction Using Machine Learning and Sentiment Analysis.” CD-MAKE (2021).
[3]	A. Garlapati, K. Garlapati, N. Malisetty, D. R. Krishna and G. Narayana, "Price Listing Predictions and Forthcoming Analysis of Airbnb," 2021 12th International Conference on Computing Communication and Networking Technologies (ICCCNT), 2021, pp. 1-7, doi: 10.1109/ICCCNT51525.2021.9579773.
[4]	Zhang, Zhihua et al. “Key Factors Affecting the Price of Airbnb Listings: A Geographically Weighted Approach.” Sustainability 9 (2017): 1635.
[5]	http://insideairbnb.com/get-the-data.html 
[6]	https://github.com/andresfmora/Airbnb-Sentiment_Analysis/blob/master/AirBNB.ipynb 
[7]	https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
[8]	https://www.kaggle.com/saurabhwadhawan/statistical-analysis-visualization-on-ny-airbnb 
