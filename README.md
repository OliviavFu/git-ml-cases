# git-ml-cases
Some machine learning cases from Kaggle.

## 01_Price_Prediction_LR_with_Uber_and_Lyft_Dataset_Boston
### Dataset size: `#693K`
Predict taxi price using linear regression, two models with different feature engineering choices are compared. 
### Main Content: 
- Exploratory data analysis
- Feature engineering and selection 
- Model comparison with different feature engineering choices 
### Model Performance: 
- explained variance score on test: __0.8954934452299009__ for model with hour and dayofweek as __numerical__ variables.
- explained variance score on test: __0.8954868395420646__ for model with hour and dayofweek as __categorical__ variables.
### Summary: 
It seems that with well enough data, the difference is trivial with how we treat the hour and dayofweek variables. If check out the coefficients, we will see that long distance ride with bad weature, especially when the passengers use
high-end products like luxuv/lux/plus/etc, will drive high taxi price which quite follows our intuitive too.  

## 02_CV_Digit_Recognizer_CNN_with_MNIST_Database
### Dataset size: `#42K handwriting images`
Build digit recognizer using convolutional neural networks, two networks with different architectures are compared. 
### Main Content: 
- Data preparation and transform
- Neural networks architechture design
- With and without batch normalization results compare 
### Model Performance:
- model WITHOUT BatchNorm: validation accuracy starts from __0.9721__ in the 1st epoch to __0.9929__ in the end.
- model WITH BatchNorm: validation accuracy starts from __0.9950__ in the 1st epoch to __0.9936__ in the end.
### Summary: 
Adam optimizer (adaptive learning rate + momentum) is used for both networks, we can see that the final model performance ends up very close for this simple dataset. BatchNorm could be helpful to achieve better performance with less epoches and maybe speed up the training process for complex CNN jobs. In the meanwhile, regular Adam without BatchNorm could work fine for many cases too. 

## 03_Fraud_Detection_Tree_Models_with_Sparkov_SimCC_Dataset
### Dataset size: `#1.3MM`
Fraud detection using different tree models and also compared imbalance learning technique and data normalization 
technique results. 
### Main Content:  
- New features derive
- WOE transformation
- Under sampling results comparison 
- Data normalization results comparison 
- Different tree models comparison 
### Model Performance:
- Decision Tree Model: f1 score on label of interest: __0.80__; precision: __0.80__; recall: __0.81__; AUC: __0.91__; 
- Random Forest: f1 score on label of interest: __0.83__; precision: __0.97__; recall: __0.72__; AUC: __0.86__;
- LightGBM: f1 score on label of interest: __0.85__; precision: __0.93__; recall: __0.78__; AUC: __0.89__; 
- XGBoost: f1 score on label of interest: __0.88__; precision: __0.95__; recall: __0.81__; AUC: __0.91__;
- CatBoost: f1 score on label of interest: __0.87__; precision: __0.94__; recall: __0.82__; AUC: __0.91__; 
### Summary: 
Random Forest has the highest precision; CatBoost has the highest recall; LightGBM runs fatest; XGBoost and CatBoost have overall a better performance and CatBoost runs faster. 

## 04_NLP_Sentiment_Classification_NB_with_IMDB_Dataset
### Dataset size: `#50K movie reviews`
Predict if a movie review is positive or negative using multinomial Naive Bayes. This is a binary sentiment classification natural language processing problem. Two methods of converting text to vectors for model training are compared. 
### Main Content:  
- Word Cloud
- Text data pre-processing
- Text vectorization 
- Models with different vectorizers comparison   
### Model Performance:
- NB with CountVectorizer: f1 score on positive class: __0.86__; precision: __0.88__; recall: __0.84__; AUC: __0.86__; 
- NB with CountVectorizer: f1 score on negative class: __0.86__; precision: __0.85__; recall: __0.88__; AUC: __0.86__; 
- NB with TfidfVectorizer: f1 score on positive class: __0.87__; precision: __0.88__; recall: __0.86__; AUC: __0.87__;
- NB with TfidfVectorizer: f1 score on negative class: __0.87__; precision: __0.86__; recall: __0.88__; AUC: __0.87__;
### Summary: 
Both CountVectorizer and TfidfVectorizer with Naive Bayes model produce decent results. TfidfVectorizer slightly 
outperforms CountVectorizer in this case. Personally speaking, if computation capacity / constraint is not an issue, I 
will prefer the TfidfVectorizer since it also takes the context of frequency in document into consideration. 

# Coming Next 
## 05_Human_Activity_Recognition_SVM_with_HAR_Database  