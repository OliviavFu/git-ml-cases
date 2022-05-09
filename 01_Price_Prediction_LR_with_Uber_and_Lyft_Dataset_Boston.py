# Kaggle Data Source: https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma
# Use the ride time and weature features to predict the ride price. 
# Some assumptions: the ride price is related to the ride time (both day in a week and time in a day), 
# weather conditions, ride distance and product type etc.. 

# Two multiple linear regression models will be compared: one treats ride time in a day as numerical varaible, 
# another treats it as categorical variable. 

# A side note: "when we say multiple regression, we mean only one dependent variable with a single distribution 
# or variance. The predictor variables are more than one. To summarise multiple refers to more than one predictor 
# variables but multivariate refers to more than one dependent variables."


#################################################################################################################
import pandas as pd
# pd.options.display.max_columns = None
# pd.options.display.max_rows = None
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Get the data
df = pd.read_csv('/kaggle/input/uber-and-lyft-dataset-boston-ma/rideshare_kaggle.csv')
df.head()
df.describe()
# Around 8% missing / NA values so remove directly. 
df.dropna(inplace=True) 

# Exploratory data analysis
df.groupby('day')['price'].mean().plot() 
df.groupby('hour')['price'].mean().plot()
# Price uptick from 12/5 - 12/9, not sure if any promotion / event during the time. 
# No specific pattern on the trend by day and two months data may be not representative enough for seasonality. 
# So day or date column is set selected as a feature for the model. 
df['date'] = df['datetime'].apply(lambda x: pd.to_datetime(x).date()) 
df.groupby('date')['price'].mean().plot(figsize=(20,3))
df['dayofweek'] = df['date'].apply(lambda x: x.weekday())
# Add 'dayofweek' column as model feature; Taxi price is high on Mon. / Wed. / Weekend.
df.groupby('dayofweek')['price'].mean().plot()

# Remove one of the duplicate columns
df[['visibility','visibility.1']].head()
df.drop('visibility.1',axis=1, inplace=True)

# Side note: 
# Multicollinearity happens when independent variables in the regression model are highly correlated to each other. 
# It makes it hard to interpret of model and also creates an overfitting problem.
# Linear Regression, Logistic Regression, KNN, and Naive Bayes algorithms are impacted by multicollinearity. 
# KNN - due to multicollinearity points gets very closer, gives incorrect results and performance will get impacted.

# Check independent variable correlations
plt.figure(figsize=(20,20))
# sns.pairplot(df,x_vars=['price'])
sns.heatmap(df.corr(),cmap='coolwarm')

# Remove high correlated numerical variables
df_corr = df.corr()
rm_list = []
for var in var_list:
    for i in range(len(var_list)):
        if var_list[i] != var and df_corr[var][i]>=0.8:
            rm_list.append(var)
            
feature_list = [x for x in df.columns if x not in rm_list]

# Manually remove categorical variables with no meaning / only 1 value / duplicate meanings. 
feature_list_sel = [#'id',
 'hour',
#  'day',
#  'month',
#  'datetime',
#  'timezone',
 'source',
 'destination',
#  'cab_type',
 'product_id',
#  'name',
#  'price',
 'distance',
# 'surge_multiplier',
#  'latitude',
#  'longitude',
 'short_summary',
#  'long_summary',
 'humidity',
 'visibility',
#  'icon',
 'pressure',
 'windBearing',
 'cloudCover',
 'uvIndex',
 'ozone',
 'moonPhase',
 'precipIntensityMax',
#  'date',
 'dayofweek']

# Get X, y for linear regression, hour and dayofweek as numerical variables. 
X = df[feature_list_sel]
y = df['price']

# Prepare another X_cat when hour and dayofweek as categorical variables. 
X_cat = X.copy()
X_cat['hour'] = X_cat['hour'].astype(str)
X_cat['dayofweek'] = X_cat['dayofweek'].astype(str)
X_cat.info()

# Convert categorical variables into dummy variables. 
cat_feats = [c for c in X.columns if X[c].dtype == object]
# cat_feats = [c for c in X.columns if isinstance(X[c][0],str)]
X_final = pd.get_dummies(X,columns=cat_feats,drop_first=True)
X_final.info()

# Side note: 
# OneHotEncoder cannot process string values ​​directly . If your nominal features are strings, then you need to first 
# map them into integers. pandas.get_dummies is kind of the opposite.

# get_dummies is the option to go with as it would give equal weightage to the categorical variables. LabelEncoder is 
# used when the categorical variables are ordinal ie if you are converting severity or ranking, then LabelEncoding 
# "High" as 2 and " low" as 1 would make sense.

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=55)
lm = LinearRegression()
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
# 0.8954934452299009 for model that with hour and dayofweek as numerical variables.
# 0.8954868395420646 for model that with hour and dayofweek as categorical variables.
# So with well enough data, the difference is trivial with how we treat the hour and dayofweek variables. 
metrics.explained_variance_score(y_test, predictions) # R^2 value is around 90%. 

# Check out the coefficients, we will see that long distance ride with bad weature, especially when the passengers use
# high-end products like luxuv/lux/plus/etc, will drive high taxi price which is quite intuitive too.  
coef_df = pd.DataFrame(lm.coef_, X_final.columns, columns = ['Coefficient'])
coef_df