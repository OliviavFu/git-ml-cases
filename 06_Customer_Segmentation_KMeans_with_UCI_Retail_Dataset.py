# Kaggle Data Source: https://www.kaggle.com/datasets/jihyeseo/online-retail-data-set-from-uci-ml-repo

# This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 
# for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many 
# customers of the company are wholesalers.

# This is an unsupervised machine learning task. We will build customer segements using k-means clustering and help 
# business owners to better understand their customer portfolio for an incentive campaign, smart / limited resource 
# allocation or even for product design and so forth. In addition, we will also compare how different level of 
# information for the task may or may not impact the model results.    


#################################################################################################################
import numpy as np 
import pandas as pd 
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# %matplotlib inline
# pip install openpyxl

# load the data
df = pd.read_excel('/kaggle/input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx')

# data check
df.head()
df.shape
df.info()
df.isnull().sum()
df.duplicated().sum()

# remove duplicates
df.drop_duplicates(inplace=True)

# remove customerID null
# The clusterinig will be done on aggregated data by customerID. So this is not the case that we could put null into 
# a group or make inference or assumptions. Inference in this case could be complicated and may lead to low accuracy 
# which could impair the clustering results. We have enough data so maybe it's better that we just simply remove null. 
df_nona = df[df['CustomerID'].isnull()==False] 
df_nona.shape

# Build driverset for clustering 
# 1 - completed payments only 
df_base_1 = df_nona[(df_nona['InvoiceNo'].str.startswith('C')!=True) & (df_nona['Quantity']>0) & (df_nona['UnitPrice']>0)]
df_cust = df_base_1.groupby(['CustomerID','Country']).count()
df_cust.groupby(['CustomerID','Country']).count().max() # check if account and country is 1:1 -> Yes
df_base_1['is_xb']=np.where(df_base_1['Country']=='United Kingdom',0,1) # add cross border transaction tag
df_base_1['Amount']=df_base_1['Quantity']*df_base_1['UnitPrice'] # add transaction amount/value
# aggregate data by account - customerID 
df_cust_a = df_base_1.groupby(['CustomerID','is_xb'])['Quantity','Amount'].sum().reset_index()
df_cust_b = df_base_1.groupby(['CustomerID','is_xb'])['InvoiceNo','StockCode'].nunique().reset_index().rename(columns={'InvoiceNo':'txn_cnt','StockCode':'item_cnt'})
df_cust_c = df_base_1.groupby(['CustomerID','is_xb'])['InvoiceDate'].max().reset_index().rename(columns={'InvoiceDate':'InvoiceDateMax'}) 
max_date = df_cust_c['InvoiceDateMax'].dt.date.max()
df_cust_c['Recency'] = (max_date - df_cust_c['InvoiceDateMax'].dt.date + datetime.timedelta(days=1)).apply(lambda x: x.days)
# check sub_df shape -> all same 
for df in [df_cust_a, df_cust_b, df_cust_c]:
    print(df.shape)
# merge all get driverset 1
# CustomerID is not used in the clustering model and sets to be table index 
df_cluster_data_1 = df_cust_a.merge(df_cust_b,on=['CustomerID','is_xb']).merge(df_cust_c.drop('InvoiceDateMax',axis=1),on=['CustomerID','is_xb']).set_index('CustomerID')

# driverset normalization 
scaler = StandardScaler()
df_cluster_scaled_1 = scaler.fit_transform(df_cluster_data_1)

# find optimal K
within_sse = []
for i in range(2,21):
    kmeans = KMeans(n_clusters=i, max_iter=50)
    kmeans.fit(df_cluster_scaled_1)
    within_sse.append(kmeans.inertia_) # check the within group SSE of elbow method 
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(df_cluster_scaled_1,cluster_labels) # silhouette score for similarity checking
    print(f'For n_cluster = {i}: The average Slhouette score is {silhouette_avg}')
    
plt.plot(within_sse, marker ='o') # We have elbow change in [5,6,7] and K=6 has the highest silhouette score 0.513996. 

# train KMeans clustering with K=6
kmeans = KMeans(n_clusters=6)
kmeans.fit(df_cluster_scaled_1)

# review the cluster results
df_cluster_data_1['clusters'] = kmeans.labels_
df_cluster = df_cluster_data_1.groupby('clusters').mean()
for col in df_cluster.columns:
    df_cluster[col].plot(kind='bar')
    plt.title(col)
    plt.show()
# By looking into the cluster feature means to understand what these derived clusters represent
# cluster 1: high mean by is_xb with medium recency and low mean for the rest 4 features so this is the "Overseas" group. 
# cluster 2: high mean by recency and low mean for the rest 5 features so this is the "Recency" group. 
# cluster 3: high mean by Quantity and Amount, not very recent and medium mean for the rest 3 features so this is the "High Value" group. 
# cluster 4: relatively low mean by everything except medium item_cnt so this seems the "Low Price Item" group.  
# cluster 5: high mean by txn_cnt and item_cnt, medium mean for Quantity and Amount and low mean for the rest 2 features so this seems the "Stocker" group. 
# cluster 6: lowest mean for almost everything except for some recency purchases so this seems the "Low Engagement" group (or are they "New" customers?). 

# Get driverset 2
# 2 - add cancelled payments info.
df_base_2 = df_nona[(df_nona['InvoiceNo'].str.startswith('C')==True) | (df_nona['Quantity']<0)] # canceled payments 
df_base_2.shape
df_base_2_a = df_base_2.groupby('CustomerID')['InvoiceNo'].nunique().reset_index().rename(columns={'InvoiceNo':'cancel_txn_cnt'}).set_index('CustomerID')
df_cluster_data_2 = df_cluster_data_1.merge(df_base_2_a,on='CustomerID', how='left').fillna(0)

# driverset normalization 
df_cluster_scaled_2 = scaler.fit_transform(df_cluster_data_2)

# find optimal K
within_sse_2 = []
for i in range(2,21):
    kmeans_2 = KMeans(n_clusters=i, max_iter=50)
    kmeans_2.fit(df_cluster_scaled_2)
    within_sse_2.append(kmeans_2.inertia_)
    cluster_labels_2 = kmeans_2.labels_
    silhouette_avg_2 = silhouette_score(df_cluster_scaled_2,cluster_labels_2)
    print(f'For n_cluster = {i}: The average Slhouette score is {silhouette_avg_2}')
    
plt.plot(within_sse_2, marker ='o') # We have elbow change in [5,6,7] and K=6 has the highest silhouette score 0.468492. 

# train KMeans clustering with K=6 
kmeans_2 = KMeans(n_clusters=6)
kmeans_2.fit(df_cluster_scaled_2)

# review the cluster results
df_cluster_data_2['clusters'] = kmeans_2.labels_
df_cluster_2 = df_cluster_data_2.groupby('clusters').mean()
for col in df_cluster_2.columns:
    df_cluster_2[col].plot(kind='bar')
    plt.title(col)
    plt.show()
# cluster 1: highest mean by recency and low mean for the rest 6 features so this is the "Recency" group. 
# cluster 2: high mean by is_xb with medium recency and low mean for the rest 5 features so this is the "Overseas" group.
# cluster 3: lowest mean for almost everything except for some recency purchases so this seems the "Low Engagement" group (or are they "New" customers?).  
# cluster 4: high mean by Quantity and Amount, not very recent and medium mean for the rest 4 features so this is the "High Value" group. 
# cluster 5: highest mean by txn_cnt, item_cnt and cancel_txn_cnt, medium mean for Quantity and Amount, no recency and relatively low mean for is_xb so this seems the "Stocker" group and they are also Canceller.  
# cluster 6: relatively low mean by everything except medium item_cnt and cancel_txn_cnt so this seems the "Low Price Item" group and they also tend to cancel. 

# Some interesting findings/thoguhts: 1) When more information passes to clustering, with same K, we have lower silhouette
# score. However this doesn't necessarily mean the new segmentation is worse. With a different driverset with more features, 
# in fact, the problem space has changed so is the silhouette score comparable cross different driversets or problem spaces? 
# I don't think so. 2) What I can imagine is that with more information, we probably have more chances to slice the space 
# with more details which may lead to closer data points in different clusters and that could lead to lower silhouette score
# if we assume to compare the scores cross problem spaces. 3) With more information, it actully results in very similar 
# segmentations in this case. And it helps us better understand the clusters/groups, in fact, I feel it segments the customers
# better with cancelled info. added and we get to know which group tends to cancel so that the store could work on that. 
# 4) Is the more information the better? Personally, I feel not necessarily. If we constantly add more and more features 
# into the driversets, I feel the segmentation interpretation might become challenging and the problem could become more 
# complicated, some dimension reduction techniques might be needed. So this might be one of those situations that we want 
# it to be not too big and not too small.      

