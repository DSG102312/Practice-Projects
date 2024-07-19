#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')


# In[3]:


from xgboost import XGBClassifier


# In[ ]:


#Red Wine Quality Prediction Project


# In[20]:


df = pd.read_csv('https://raw.githubusercontent.com/FlipRoboTechnologies/ML-Datasets/main/Red%20Wine/winequality-red.csv')
df


# In[23]:


#exploring the type of data present in each of the columns present in the dataset.
df.info()


# In[24]:


#To see the count, mean, standard deviation, minimum, maximum and inter quantile values of the dataset.
df.describe()


# In[25]:


#To see the count, mean, standard deviation, minimum, maximum and inter quantile values of the dataset and explore the descriptive statistical measures of the dataset.
df.describe().T


# In[ ]:


#Exploratory Data Analysis
#EDA is an approach to analysing the data using visual techniques. It is used to discover trends, and patterns, or to check assumptions with the help of statistical summaries and graphical representations.


# In[61]:


#To see total rows and columns present in our datasets.
df.shape


# In[62]:


# To check the number of null values in the dataset columns wise.
df.isnull().sum()


# In[26]:


#imputing the missing values by means as the data present in the different columns are continuous values.

for col in df.columns:
  if df[col].isnull().sum() > 0:
    df[col] = df[col].fillna(df[col].mean())
 
df.isnull().sum().sum()


# In[27]:


#Drawing the histogram to visualise the distribution of the data with continuous values in the columns of the dataset.

df.hist(bins=20, figsize=(10, 10))
plt.show()


# In[28]:


#Drawing the count plot to visualise the number data for each quality of wine.

plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# In[29]:


#There are times the data provided to us contains redundant features they do not help with increasing the model’s performance that is why we remove them before using them to train our model.

plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()


# In[30]:


#From the above heat map we can conclude that the ‘total sulphur dioxide’ & ‘free sulphur dioxide‘ are highly correlated features so, we will remove them.

df = df.drop('total sulfur dioxide', axis=1)


# In[31]:


#Model Development

#preparing the data for training and splitting it into training and validation data so, that we can select which model’s performance is best as per the use case. We will train some of the state of the art machine learning classification models and then select best out of them using validation data.

df['best quality'] = [1 if x > 5 else 0 for x in df.quality]


# In[32]:


#We have a column with object data type as well as we replace it with the 0 and 1 as there are only two categories.

df.replace({'white': 1, 'red': 0}, inplace=True)


# In[33]:


#After segregating features and the target variable from the dataset we will split it into 80:20 ratio for model selection.

features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']
 
xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40)
 
xtrain.shape, xtest.shape


# In[34]:


#Normalising the data before training help us to achieve stable and fast training of the model.

norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)


# In[35]:


from sklearn.preprocessing import StandardScaler
s =StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.fit_transform(X_test)


# In[36]:


#As the data has been prepared completely let’s train some state of the art machine learning model on it

models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
 
for i in range(3):
    models[i].fit(xtrain, ytrain)
 
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(
        ytest, models[i].predict(xtest)))
    print()


# In[30]:


from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# In[31]:


#Model Evaluation
#From the above accuracies we can say that Logistic Regression and SVC() classifier performing better on the validation data with less difference between the validation and training data. Let’s plot the confusion matrix as well for the validation data using the Logistic Regression model.

class_names = df.columns
metrics.plot_confusion_matrix(models[1], X_test, Y_test,cmap='mako')
plt.show()


# In[71]:


# print the classification report for the best performing model.
print(metrics.classification_report(ytest,
                                    models[1].predict(xtest)))

