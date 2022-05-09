# Kaggle Data Source: https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTrain.csv

# We will build a fraud detection model using different tree algorithms. Results applying imbalanced learning 
# technique and data normalization techique are also compared during the training process. 


#################################################################################################################
import pandas as pd
import numpy as np
import math
from datetime import date
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm import LGBMClassifier
import xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
pd.options.display.max_columns = None

# Load the data
# This is simulated data using Sparkov so no "NA" values   
df_train = pd.read_csv('/kaggle/input/fraud-detection/fraudTrain.csv')

# Feature Engineering 
df_train.info()
# New features derived
df_train['pmt_hour'] = pd.to_datetime(df_train['trans_date_trans_time']).dt.hour
df_train['pmt_window'] = np.where(df_train['pmt_hour'].between(6,10),'morning',
                                 np.where(df_train['pmt_hour'].between(11,12),'noon', 
                                 np.where(df_train['pmt_hour'].between(13,17),'afternoon',
                                 np.where(df_train['pmt_hour'].between(18,22),'evening','midnight')))) # pmt_window: payment time block during the day

df_train['buyer'] = df_train['first'] + ' '+df_train['last']
df_train['s_r_link'] = df_train['merchant'] + ' '+df_train['buyer']
right_df = df_train.groupby('s_r_link')['amt'].count().to_frame().rename(columns={'amt':'s_r_pmt_hist_cnt'})
df_train = df_train.merge(right_df, how='left', right_index = True, left_on = 's_r_link') # s_r_pmt_hist_cnt: buyer and merchant payment counts

df_train['zip'] = df_train['zip'].astype('str') 

df_train['dob_year'] = pd.to_datetime(df_train['dob']).dt.year
df_train['age'] = df_train['dob_year'].apply(lambda x: date.today().year - x) # age: buyer age

df_train['distance'] = np.linalg.norm(df_train[['lat','long']].values - df_train[['merch_lat','merch_long']].values, axis = 1) # distance: buyer and merchant location distance

# label and selected features
selected_list = ['merchant', 'category','amt', 'gender', 'city', 'state', 'zip',
       'lat', 'long', 'city_pop', 'job', 'merch_lat', 'merch_long','pmt_hour', 'pmt_window',
       's_r_pmt_hist_cnt', 'age', 'distance','is_fraud']
df_selected = df_train[selected_list]

# WOE transformation for categorical features
# get all categorical features
cat_feat = []
for x in df_selected.columns:
    if df_selected[x].dtypes == object:
        cat_feat.append(x)
        
# define WOE function 
def cat_woe_transform(var_list, df, target, count_var):
    '''transform categorical variable into woe for sklearn tree algorithms
       target: varaible of interest 
       count_var: number count by selected variable
    '''
    df_transform = df
    for var in var_list:
        df_a = df[df[target]==1].groupby(var).count()[count_var].to_frame().rename(columns={count_var:'event_cnt'})
        df_b = df[df[target]!=1].groupby(var).count()[count_var].to_frame().rename(columns={count_var:'non_event_cnt'})
        df_c = df_a.join(df_b)
        df_c['event_pct']=df_c['event_cnt'].apply(lambda x: (x+0.5)/df_c['event_cnt'].sum())
        df_c['non_event_pct']=df_c['non_event_cnt'].apply(lambda x: (x+0.5)/df_c['non_event_cnt'].sum())
        df_c[var+'_woe']= (df_c['non_event_pct']/df_c['event_pct']).apply(lambda x: math.log(x))
        df_transform = pd.merge(df_transform,df_c[var+'_woe'].reset_index(),how='left',on=var)
        df_transform.drop(var, axis=1, inplace = True)
    return df_transform.dropna() # drop NaN values from WOE transformation 
    # fill NaN is also tested but the model performance is poor so no fill-in applied.  

# transform data
df_woe = cat_woe_transform(cat_feat, df_selected, 'is_fraud', count_var='amt')

# Training Decision Tree Model 
# Get X and y 
X = df_woe.drop('is_fraud',axis=1)
y = df_woe['is_fraud']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55)

# train the model 
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)

# performance review
print(classification_report(y_test,dtree_pred)) # f1 score on label of interest: 0.8; precision: 0.8; recall: 0.81
print('\n')
print(confusion_matrix(y_test,dtree_pred))
print('\n')
print(f'Test AUC Score: {roc_auc_score(y_test,dtree_pred)}') # AUC: 0.91

# The original dataset is imbalanced with 0.6% fraud payments. Here uses under sampling to increase the fraud rate 
# for training dataset X_train to 10% and see if any improvements on model performance. 
undersample = RandomUnderSampler(sampling_strategy=0.1)
X_under, y_under = undersample.fit_resample(X_train, y_train)
print(Counter(y_under)) # check the under sample results

# train the model with under sampled data
dtree_under = DecisionTreeClassifier()
dtree_under.fit(X_under, y_under)
dtree_pred_under = dtree_under.predict(X_test)

# performance review
print(classification_report(y_test,dtree_pred_under)) # f1 score on label of interest: 0.53; precision: 0.37; recall: 0.91
print('\n')
print(confusion_matrix(y_test,dtree_pred_under))
print('\n')
print(f'Test AUC Score: {roc_auc_score(y_test,dtree_pred_under)}') # AUC: 0.95

# With under sampling, model recall is higher. It's probably due to more fraud patterns are learned with a higher fraud 
# data rate while less good samples in the training process leads to a lower precision on test, which means the model has
# less information on good to distinguish it from bad. 
# Despite a higher AUC score, the model without under sampling gives us a better performance on fraud (label of interest) 
# population, so model without under sampling will be applied in the following sections. 

# In honor of the idea that in general, it's good to do data normalization for machie learning. 
# Let's check if data normalization would help improve the model performance. 
def normalize(df):
    '''min-max scaling'''
    df_norm = df.copy()
    for c in df_norm.columns:
        var_min = df_norm[c].min()
        var_max = df_norm[c].max()
        df_norm[c] = df_norm[c].apply(lambda x: (x-var_min)/(var_max - var_min + 10**(-10)))
    return df_norm

# get normalized data, it takes some time. 
X_norm = normalize(X_train)
X_test_norm = normalize(X_test)  

# train the model with normalization 
dtree_norm = DecisionTreeClassifier()
dtree_norm.fit(X_norm, y_train)
dtree_pred_norm = dtree_norm.predict(X_test_norm)

# performance review
print(classification_report(y_test,dtree_pred_norm)) # f1 score on label of interest: 0.78; precision: 0.76; recall: 0.81
print('\n')
print(confusion_matrix(y_test,dtree_pred_norm))
print('\n')
print(f'Test AUC Score: {roc_auc_score(y_test,dtree_pred_norm)}') # AUC: 0.90

# Train with normalized data gets similiar results as without normalization. 
# So normalization is not applied to this case either. 

# Now let's check which tree algorithm produces the best predictions. 

# Random Forest, this takes a while 
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)

rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred)) # f1 score on label of interest: 0.83; precision: 0.97; recall: 0.72
print('\n')
print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(f'Test AUC Score: {roc_auc_score(y_test,rfc_pred)}') # AUC: 0.86

# The precision on label of interest from RF is very high. Maybe this reflects the learning ability of ensemble models.  

# LightGBM, this is fast! 
lgbm_2 = LGBMClassifier(n_estimators=500, lambda_l2=0.1) # boosting_type is 'gbdt' by default 
lgbm_2.fit(X_train, y_train)

lgbm_pred_2=lgbm_2.predict(X_test)
print(classification_report(y_test,lgbm_pred_2)) # f1 score on label of interest: 0.85; precision: 0.93; recall: 0.78
print('\n')
print(confusion_matrix(y_test,lgbm_pred_2))
print('\n')
print(f'Test AUC Score: {roc_auc_score(y_test,lgbm_pred_2)}') # AUC: 0.89

# Side note: 
# The key difference in speed is because XGBoost (depth-first: level-wise) split the tree nodes one level at a time 
# and LightGBM (best-first: leaf-wise) does that one node at a time.
# It is not advisable to use LGBM on small datasets. Light GBM is sensitive to overfitting and can easily overfit 
# small data.

# XGBoost, this is not as fast as LightGBM but still faster than previous models. 
xgbm = XGBClassifier(n_estimators=500, learning_rate=0.05).fit(X_train, y_train)
xgbm_pred = xgbm.predict(X_test)

print(classification_report(y_test,xgbm_pred)) # f1 score on label of interest: 0.88; precision: 0.95; recall: 0.81
print('\n')
print(confusion_matrix(y_test,xgbm_pred))
print('\n')
print(f'Test AUC Score: {roc_auc_score(y_test,xgbm_pred)}') # AUC: 0.91

# CatBoost, this seems faster than XGBoost but not as fast as LightGBM
catb = CatBoostClassifier(verbose=0, n_estimators=500).fit(X_train, y_train)
catb_pred = catb.predict(X_test)

print(classification_report(y_test,catb_pred)) # f1 score on label of interest: 0.87; precision: 0.94; recall: 0.82
print('\n')
print(confusion_matrix(y_test,catb_pred))
print('\n')
print(f'Test AUC Score: {roc_auc_score(y_test,catb_pred)}') # AUC: 0.91

# Side note:
# The primary benefit of the CatBoost (in addition to computational speed improvements) is support for 
# categorical input variables. This gives the library its name CatBoost for “Category Gradient Boosting.”
 