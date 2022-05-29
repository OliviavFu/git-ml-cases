# Kaggle Data Source: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones

# The Human Activity Recognition database was built from the recordings of 30 study participants performing 
# activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors. 
# The objective is to classify activities into one of the six activities performed.

# This is a multiclass classification problem. We will predict human activity from sensor data using support vector 
# machine. Model performances are compared with different dimension reduction techniques.       


#################################################################################################################
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
# %matplotlib inline

# load the data
df_train = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/train.csv')
df_test = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/test.csv')

# check missing data
df_train.isna().sum()
df_test.isna().sum()
df_train.isnull().values.any() # no missing 
df_test.isnull().values.any() # no missing 

# check the data
df_{}.shape
df_{}.describe()
df_{}.columns

# label encoding
le = preprocessing.LabelEncoder()
df_train['Activity_encode'] = le.fit_transform(df_train['Activity'])
df_test['Activity_encode'] = le.fit_transform(df_test['Activity'])

# remove low variance features
df_train.var().describe() # 0.01 is selected as remove benchmark based off the feature variance distribution 
df_var = df_train.var().to_frame('variance')
# df_var[df_var['variance'] <= 0.01].count()
low_var_list = df_var[df_var['variance'] <= 0.01].index.tolist()
# len(low_var_list)
remove_list = low_var_list + ['subject','Activity'] # remove subject and old Activity label too
# len(remove_list)
df_train.drop(remove_list, axis = 1, inplace = True)
df_test.drop(remove_list, axis = 1, inplace = True)

# Get X, y
X_train = df_train.drop('Activity_encode',axis=1)
y_train = df_train['Activity_encode']
X_test = df_test.drop('Activity_encode',axis=1)
y_test = df_test['Activity_encode']

# train SVM with default settings
svc_0 = SVC()
svc_0.fit(X_train,y_train)
y_pred_0 = svc_0.predict(X_test)
print(f'accuracy score: {accuracy_score(y_test,y_pred_0)}') # 95% overall accuracy 
print('\n')
print(f'confusion matrix:\n {confusion_matrix(y_test,y_pred_0)}')
print('\n')
print(f'classification report:\n {classification_report(y_test,y_pred_0)}') # 92%+ f1 score on all label categories

# train SVM with CV hyperparameter tuning
param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
svc = GridSearchCV(SVC(),param_grid,verbose=3)
svc.fit(X_train, y_train) # this will take some time
svc.best_estimator_
svc.best_score_
y_pred = svc.predict(X_test)
print(f'accuracy score: {accuracy_score(y_test,y_pred)}') # 96.47% overall accuracy 
print('\n')
print(f'confusion matrix:\n {confusion_matrix(y_test,y_pred)}')
print('\n')
print(f'classification report:\n {classification_report(y_test,y_pred)}') # 94%+ f1 score on all label categories

# SVM with different dimension reduction techniques performance comparison 
# with principal component analysis (PCA)
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X_train) # normalize the data for PCA
X_scaled_test = scaler.fit_transform(X_test)

pca_sel = PCA(n_components=len(X_train.columns))
pca_sel_data = pca_sel.fit(X_scaled)
# pca_sel_data.explained_variance_ratio_
plt.plot(np.cumsum(pca_sel_data.explained_variance_ratio_))
plt.xlabel('number of pca')
plt.ylabel('cumulative explained variance') # check desired # of component to explain the data variance 

pca = PCA(n_components=200) # 200 components are selected 
X_pca = pca.fit(X_scaled)
X_pca_train = X_pca.transform(X_scaled)
X_pca_test = X_pca.transform(X_scaled_test) # transform test data into pca with train data results

svc_pca = SVC(C=1000, gamma=0.01)
svc_pca.fit(X_pca_train,y_train) # fit model with pca 
y_pca_pred = svc_pca.predict(X_pca_test)
print(f'accuracy score: {accuracy_score(y_test,y_pca_pred)}') # 90% overall accuracy 
print('\n')
print(f'confusion matrix:\n {confusion_matrix(y_test,y_pca_pred)}')
print('\n')
print(f'classification report:\n {classification_report(y_test,y_pca_pred)}') # 89%+ f1 score on most of the label categories and only 76% f1 score on label 4. 

# Note: Interestingly, the model performance decreases with PCA. It turns out that using PCA can lose some spatial 
# information which could can impair SVM for classification, so you might want to retain more dimensions so that SVM 
# retains more information. Anyways, let's see if other dimension reduction techniques could improve SVM performance or 
# it's better without any dimension reduction applied?!

# with independent component analysis (ICA) 
ica = FastICA(n_components=200)
X_ica = ica.fit(X_scaled)
X_ica_train = X_ica.transform(X_scaled)
X_ica_test = X_ica.transform(X_scaled_test)

svc_ica = SVC(C=1000,gamma=0.01)
svc_ica.fit(X_ica_train,y_train)
y_ica_pred = svc_ica.predict(X_ica_test)
print(f'accuracy score: {accuracy_score(y_test,y_ica_pred)}') # 96.23% overall accuracy 
print('\n')
print(f'confusion matrix:\n {confusion_matrix(y_test,y_ica_pred)}')
print('\n')
print(f'classification report:\n {classification_report(y_test,y_ica_pred)}')
# 91%+ f1 score on all the label categories; and SVM with ICA has a better performance on category 3, 4, 5 while the pure
# SVM does a better job predicting category 1, 2. It could be that the ICA reduces the dimensions and at the same time 
# keeps crucial information with component independence considered. So with ICA, we have reduced the feature dimensions 
# from 500+ to 200 and maintained a decent model performance. 
  
