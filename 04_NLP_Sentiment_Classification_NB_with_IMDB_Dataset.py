# Kaggle Data Source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# IMDB dataset having 50K movie reviews for natural language processing or Text analytics.

# Predict if a movie review is positive or negative using multinomial Naive Bayes. This is a binary sentiment 
# classification natural language processing problem. Two methods of converting text to vectors are compared. 


#################################################################################################################
import pandas as pd
import string
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# load the data
df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

# na check
df.info()
df.isna().sum()

# data visualization - wordcloud
wordcloud = WordCloud().generate(' '.join(df[df['sentiment']==1]['review'].apply(lambda x: BeautifulSoup(x,'html.parser').get_text())))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.axis('off')

wordcloud = WordCloud().generate(' '.join(df[df['sentiment']==0]['review'].apply(lambda x: BeautifulSoup(x,'html.parser').get_text())))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.axis('off')

# clean the text 
def text_processor(text):
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text() # remove html syntax
    text_nopunc = [char for char in text if char not in string.punctuation] # remove punctuation 
    text_nopunc = ''.join(text_nopunc)
    text_bow = [word for word in text_nopunc.split() if word.lower() not in stopwords.words('english')] # remove stopwords
    return text_bow

# Get X, y
df['sentiment'].replace('positive', 1, inplace=True)
df['sentiment'].replace('negative', 0, inplace=True)
X = df['review']
y = df['sentiment']

# Convert text X into vectors for machine learning
bow_count = CountVectorizer(analyzer=text_processor) # this takes a while
bow_tfidf = TfidfVectorizer(analyzer=text_processor) # this takes a while
X_count = bow_count.fit_transform(X)
X_tfidf = bow_tfidf.fit_transform(X)

# Side note 1: 
# TfidfVectorizer is equivalent to CountVectorizer followed by TfidfTransformer.
# In CountVectorizer we only count the number of times a word appears in the document which results in biasing in favour 
# of most frequent words. this ends up in ignoring rare words which could have helped is in processing our data more 
# efficiently.
# In TfidfVectorizer we consider overall document weightage of a word. It helps us in dealing with most frequent words. 
# Using it we can penalize them. TfidfVectorizer weights the word counts by a measure of how often they appear in the 
# documents.

# Side note 2: 
# TF-IDF is a word-document mapping (with some normalization). It ignore the order of words and gives nxm 
# matrix (or mxn depending on implementation) where n is number of words in the vocabulary and m is number of documents. 
# Word2Vec on the other hand gives a unique vector for each word based on the words appearing around the particular word. 
# TF-IDF is obtained from straightforward linear algebra. Word2Vec is obtained from the hidden layer of a two layered 
# neural network. TF-IDF can be used either for assigning vectors to words or to documents. Word2Vec can be directly 
# used to assign vector to a word but to get the vector representation of a document further processing is needed.
# Unlike TF-IDF, Word2Vec takes into account placement of words in a document (to some extent). 

# train test split
X_train_cnt, X_test_cnt, y_train, y_test = train_test_split(X_count, y, test_size=0.3, random_state=55)
X_train_tf, X_test_tf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=55)

# model train with CountVectorizer
mnb = MultinomialNB()
mnb.fit(X_train_cnt,y_train)
y_pred_cnt = mnb.predict(X_test_cnt)

print(classification_report(y_test, y_pred_cnt))
print('\n')
print(confusion_matrix(y_test, y_pred_cnt))
print('\n')
print(f'model auc score is {roc_auc_score(y_test, y_pred_cnt)}')
# f1 score on positive class: 0.86; precision: 0.88; recall: 0.84; AUC: 0.86;
# f1 score on negative class: 0.86; precision: 0.85; recall: 0.88; AUC: 0.86;

# model train with TfidfVectorizer 
mnb_2 = MultinomialNB()
mnb_2.fit(X_train_tf,y_train)
y_pred_tf = mnb_2.predict(X_test_tf)

print(classification_report(y_test, y_pred_tf))
print('\n')
print(confusion_matrix(y_test, y_pred_tf))
print('\n')
print(f'model auc score is {roc_auc_score(y_test, y_pred_tf)}')
# f1 score on positive class: 0.87; precision: 0.88; recall: 0.86; AUC: 0.87;
# f1 score on negative class: 0.87; precision: 0.86; recall: 0.88; AUC: 0.87;

# Both CountVectorizer and TfidfVectorizer with Naive Bayes model produce decent results. TfidfVectorizer slightly 
# outperforms CountVectorizer in this case. Personally speaking, if computation capacity / constraint is not an issue, I 
# will prefer the TfidfVectorizer since it also takes the context of frequency in document into consideration.      