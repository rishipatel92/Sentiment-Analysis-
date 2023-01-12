#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize


# In[30]:


from bs4 import BeautifulSoup


# In[31]:


df = pd.read_csv("IMDB Dataset.csv")


# In[32]:


df


# # Refining the Text Data

# In[33]:


# Removing HTML Strings
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Removing Square Brackets 
def remove_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Removing Special Character
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text


def combine_all(text):
    text = strip_html(text)
    text = remove_square_brackets(text)
    text = remove_special_characters(text)
    return text

df['review']=df['review'].apply(combine_all)


# # Text Processing

# In[34]:


#Stemming the text and Applying function on review
def stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text


df['review']=df['review'].apply(stemmer)


# In[35]:


#set stopwords to english
stop=set(stopwords.words('english'))
print(stop)


# In[36]:


#Tokenization of text
tokenizer=ToktokTokenizer()

#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')


# In[37]:


#removing the stopwords and Applying function on review

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


df['review']=df['review'].apply(remove_stopwords)


# In[38]:


x = df.drop(['sentiment'],axis=1)


# In[39]:


y = df['sentiment']


# # Spliting the Data for Training and Testing the Model

# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


training_data, testing_data = train_test_split(y, test_size=0.2, random_state=25)


# In[90]:


# Here we are using Tfidf Vectorizer because it not only focuses on the frequency
# of words present in the corpus but also provides the importance of the words.

from sklearn.feature_extraction.text import TfidfVectorizer


# In[234]:


tv=TfidfVectorizer(min_df=0,max_df=0.5,use_idf=False,ngram_range=(1,2),norm='l1')


# In[235]:


train_reviews=tv.fit_transform(training_data)
test_reviews=tv.transform(testing_data)


# In[236]:


from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()


# In[237]:


#Using Label Binarizer to Encode the Sentiment Column 
sentiment_data=lb.fit_transform(y)


# In[238]:


train_sentiments=sentiment_data[:40000]
test_sentiments=sentiment_data[40000:]


# In[239]:


from sklearn.linear_model import LogisticRegression


# In[240]:


lr = LogisticRegression()


# In[241]:


model=lr.fit(train_reviews,train_sentiments)


# In[242]:


y_pred = model.predict(test_reviews)


# In[243]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix


# In[244]:


accuracy_score(test_sentiments,y_pred)


# In[245]:


plot_confusion_matrix(model,test_reviews,test_sentiments)


# In[246]:


mnb_bow_report=classification_report(test_sentiments,y_pred,target_names=['Positive','Negative'])
print(mnb_bow_report)


# In[247]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[248]:


model_nb = nb.fit(train_reviews,train_sentiments)


# In[249]:


y_pred_nb = model_nb.predict(test_reviews)


# In[250]:


accuracy_score(test_sentiments,y_pred_nb)


# In[251]:


plot_confusion_matrix(model_nb,test_reviews,test_sentiments)


# In[ ]:




