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

## 05_Human_Activity_Recognition_SVM_with_HAR_Database  
### Dataset size: `#10K data points` * 561 human activity derived features  
This is a multiclass classification problem. We will predict human activity from sensor data using support vector machine. Model performances are compared with different dimension reduction techniques.    
### Main Content:  
- Data preparation 
- SVM hyperparameter tuning with cross validation (grid search)  
- SVM model with PCA 
- SVM model with ICA   
### Model Performance:
- SVM with default settings: overall test accuracy score: __0.95__; test f1 score on all 6 label activities: __0.92+__; 
- SVM with tuned parameters: overall test accuracy score: __0.9647__; test f1 score on all 6 label activities: __0.94+__; 
- SVM with PCA: overall test accuracy score: __0.90__; test f1 score on 5/6 label activities: __0.89+__ and __0.76__ on label activity __4__; 
- SVM with ICA: overall test accuracy score: __0.9623__; test f1 score on all 6 label activities: __0.91+__;  
### Summary: 
SVM does a good job classifying human activities with the dataset. Interestingly, PCA reduces SVM performance. It turns out that using PCA can lose some spatial information which could impair SVM for classification, so you might want to retain more dimensions so that SVM retains more information. On the other hand, ICA has reduced the problem feature dimensions from 500+ to 200 and maintained a decent model performance. In fact, SVM with ICA even has a better performance predicting activity __3__, __4__, __5__ when pure SVM does a better job with activity __1__, __2__. It could be that the ICA keeps crucial information when reducing dimensionality with component independence considered. So it looks like that for activity __3 - 5__, there is sort of generalization information could be extracted for classification while for activity __1 - 2__, the patterns might be hidden more under the original / raw dataset.   

## 06_Customer_Segmentation_KMeans_with_UCI_Retail_Dataset  
### Dataset size: `#542K data points`
This is an unsupervised machine learning task. We will build customer segements using k-means clustering and help business owners to better understand their customer portfolio for incentive campaign, smart / limited resource allocation or even for product design and so forth. In addition, we will also compare how different level of information for the task may or may not impact the model results.    
### Main Content:  
- Driversets cleaning and derive 
- Find optimal K by elbow method and average silhouette_score
- Review clusters with base driverset 
- Review clusters with a second driverset (base driverset + payment cancel info.)    
### Model Performance:
- base driverset #1: __K=6__ and the average silhouette score: __0.513996__; 
- #1 cluster 1: high mean by is_xb with medium recency and low mean for the rest so this is the __Overseas__ group. 
- #1 cluster 2: high mean by recency and low mean for the rest 5 features so this is the __Recency__ group. 
- #1 cluster 3: high mean by Quantity and Amount, not very recent and medium mean for the rest so this is the __High Value__ group. 
- #1 cluster 4: relatively low mean by everything except medium item_cnt so this seems the __Low Price Item__ group.  
- #1 cluster 5: high mean by txn_cnt and item_cnt, medium mean for Quantity and Amount and low mean for the rest so this seems the __Stocker__ group. 
- #1 cluster 6: lowest mean for almost everything except for some recency purchases so this seems the __Low Engagement__ group (or are they "New" customers?). 
- the second driverset #2: __K=6__ and the average silhouette score: __0.468492__; 
- #2 cluster 1: highest mean by recency and low mean for the rest so this is the __Recency__ group. 
- #2 cluster 2: high mean by is_xb with medium recency and low mean for the rest so this is the __Overseas__ group.
- #2 cluster 3: lowest mean for almost everything except for some recency purchases so this seems the __Low Engagement__ group (or are they "New" customers?).  
- #2 cluster 4: high mean by Quantity and Amount, not very recent and medium mean for the rest so this is the __High Value__ group. 
- #2 cluster 5: highest mean by txn_cnt, item_cnt and cancel_txn_cnt, medium mean for Quantity and Amount, no recency and relatively low mean for is_xb so this seems the __Stocker__ group and they are also "Canceller".  
- #2 cluster 6: relatively low mean by everything except medium item_cnt and cancel_txn_cnt so this seems the __Low Price Item__ group and they also tend to cancel. 
### Summary: 
Some interesting findings/thoguhts: 1) When more information passes to clustering, with same K, we have lower silhouette score. However this doesn't necessarily mean the new segmentation is worse. With a different driverset with more features, in fact, the problem space has changed so is the silhouette score comparable cross different driversets or problem spaces? I don't think so. 2) What I can imagine is that with more information, we probably have more chances to slice the space with more details which may lead to closer data points in different clusters and that could lead to lower silhouette score if we assume to compare the scores cross problem spaces. 3) With more information, it actully results in very similar segmentations in this case. And it helps us better understand the clusters/groups, in fact, I feel it segments the customers better with cancelled info. added and we get to know which group tends to cancel so that the store could work on that. 4) Is the more information the better? Personally, I feel not necessarily. If we constantly add more and more features into the driversets, I feel the segmentation interpretation might become challenging and the problem could become more complicated, some dimension reduction techniques might be needed. So this might be one of those situations that we want it to be not too big and not too small.    

# Coming Next 
## 07_Diabetes_Prediction_KNN_with_Diabetes_Database
## 08 Autoencoder Application...