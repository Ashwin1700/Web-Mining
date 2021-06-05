#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import re


# In[2]:


dataset = pd.read_csv(r'./IMDB.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
#X[1][0].replace(r'\\','')
#str(X[1][0]).replace('\\','e')


# In[3]:


X = pd.DataFrame(X,columns =['reviews'])
y = pd.DataFrame(y,columns = ['Score'])
pd.options.display.max_colwidth=4000

X.iloc[0][0]


# In[4]:


y['Score'].value_counts()


# In[5]:


for review_no in range(len(X['reviews'])):
    j =  re.sub(r'(.br./>)','',X.iloc[review_no][0])
    j = re.sub("[^a-zA-Z',\.() ]+", '', j)
    X.iloc[review_no] = j
    


# In[6]:


X.iloc[8][0]
y


# In[7]:


y = dataset.iloc[:, 1].values
y = pd.DataFrame(y,columns = ['Score'])

y


# In[8]:


# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.
# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.
def partition(x):
    if x == 'positive':
        return 1
    return 0

#changing reviews with score less than 3 to be positive and vice-versa
actualScore = y['Score']
positiveNegative = actualScore.map(partition) 
positiveNegative


# In[9]:


y['Score'] = positiveNegative
print("Number of data points in our data", y.shape)
y


# In[10]:


X1 = X.iloc[:,0].values
y1 = y.iloc[:,0].values


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)
y_train


# In[14]:


from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.2, random_state = 0)

t1 = X_train1
t1 = t1.assign(Score = y_train1['Score'])
#t1 =t1.reset_index(drop = True)
t2 = X_test1
t2 = t2.assign(Score = y_test1['Score'])


# In[15]:


t1.to_csv('reviews_train',index=False)
t2.to_csv('reviews_test',index=False)


# In[27]:


'''X_train_list = list(X_train['reviews'])
len(X_train_list)
X_test_list = list(X_test['reviews'])'''
type(X_train)


# In[17]:


counter = CountVectorizer()
counter.fit(X_train)


# In[18]:


#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(X_train)#transform the training data
counts_test = counter.transform(X_test)#transform the testing data


# In[19]:


counts_test
y_train


# In[20]:


#train classifier
model1 = DecisionTreeClassifier()
model2 = MultinomialNB()
model3=LogisticRegression(solver='liblinear')


# In[21]:


predictors=[('nb',model1),('dt',model2),('lreg',model3)]


# In[22]:


VT=VotingClassifier(predictors)


# In[23]:


VT.fit(counts_train,y_train)

#use hard voting to predict (majority voting)
VT.fit(counts_train,y_train)


# In[24]:


pred=VT.predict(counts_test)

#print accuracy
print (accuracy_score(pred,y_test))


# In[25]:


#X_test


# In[26]:


output = pd.DataFrame({'Reviews': X_test ,
                       'Score': pred})
output.to_csv('output.csv', index=False)


# In[ ]:





# In[ ]:




