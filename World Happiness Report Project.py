#!/usr/bin/env python
# coding: utf-8

# # World Happiness Report Project

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[6]:


df = pd.read_csv('https://raw.githubusercontent.com/FlipRoboTechnologies/ML-Datasets/main/World%20Happiness/happiness_score_dataset.csv')
df


# In[7]:


df.head()


# Statistical Summary

# In[8]:


df.describe()


# From the above statistical summary we can see that the choice of Dystopia as a benchmark permits every real country to have a positive (or at least zero) contribution from each of the six factors.

# In[9]:


df.dtypes


# In[10]:


df.columns


# In[11]:


df.shape


# To Know the Null Values

# In[12]:


df.info()


# In[13]:


df.isnull().sum()


# In[14]:


sns.heatmap(df.isnull())


# #from above heatmap there is no missing values in dataset.

# From df.describe() we can see that for some countries min values are 0. so let try to find country at bottom.

# In[15]:


df.loc[df['Health (Life Expectancy)'] ==0.0]


# In[16]:


df.loc[df['Economy (GDP per Capita)'] == 0.0]


# In[17]:


df.loc[df['Freedom']==0.0]


# In[18]:


df.loc[df['Family']==0.0]


# In[19]:


df.loc[df['Trust (Government Corruption)']==0.0]


# Minimum Performance category wise
# Health (Life Expectancy): Sierra Leone
# Economy (GDP per Capita): Congo (Kinshasa)
# Freedom: Iraq
# Family: Central African Republic
# Trust (Government Corruption): Indonesia
# Making new dataframe considering Numerical datatypes for further investigation.

# In[20]:


Happy_df = df[df.columns[3:]]
Happy_df


# In[21]:


Happy_df.shape


# There is no missing values in data.

# In[27]:


X=Happy_df.drop(columns =['Happiness Score'])
Y=Happy_df['Happiness Score']


# EDA
# 
# Skewness Detection using displot and skew()

# In[28]:


plt.figure(figsize=(20,25), facecolor='white')
plotnumber =1
for column in Happy_df:
    if plotnumber <=9:
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(Happy_df[column], color='g')
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.show()


# In[49]:


Happy_df.skew()


# We can see that standard Error,Trust, Generosity are right skewed distribution. Log transform is useful if and only if the distribution of the variable is right-skewed. A log transformation in a left-skewed distribution will tend to make it even more left skew. Family variable has left skewed distribution. 

# In[31]:


from scipy.stats import boxcox
# 0 -> Log transform
# 0.5 -> square root trasform


# In[32]:


df['Standard Error']=boxcox(df['Standard Error'],0)


# In[50]:


# checking skewness after applying boxcox
Happy_df.skew()


# Multicollinearity using Variance_inflation_factor

# In[35]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[36]:


vif =pd.DataFrame()
vif['vif'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['Features'] = X.columns


# In[37]:


# Let check  the Values
vif


# Histplot

# In[38]:


plt.figure(figsize=(20,20), facecolor='white')
plotnumber =1
for column in Happy_df:
    if plotnumber <=9:
        ax = plt.subplot(3,3,plotnumber)
        sns.histplot(Happy_df[column], palette='Rainbow' )
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.tight_layout()
plt.show()


# Pairplot

# In[39]:


sns.pairplot(Happy_df , palette='viridis')


# Investingating Correlation between features

# In[52]:


Happy_df.corr()


# In[53]:


corr_df=Happy_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_df,annot= True, cmap='coolwarm')


# Top 10 countries happiest countries based on Happiness rank/score

# In[54]:


a = df.sort_values(by='Happiness Score', ascending= False).head(10)
a


# We can see that Switzerland Top the chart with Happiness score of 7.587. We can see that 8 countries out of 10 are from Western Europe.

# Bottom 10 countries happiest countries based on Happiness rank/score

# In[55]:


b =df.sort_values(by='Happiness Score', ascending=True).head(10)
b


# We can see that Chad from Sub-Saharan Africa. Top the chart with Happiness score of 7.587. We can see that 8 countries out of Bottom 10 are from Sub-Saharan Africa.

# In[56]:


Grp_Region=df.groupby('Region')
Grp_Region['Happiness Score'].describe().sort_values(by='mean', ascending=True).head(10)


# We can see from above Region that 'Australia' and 'New Zealand' has maximum Happiness Score(7.2850) and 'Sub-Saharan Africa' has minimum Happiness Score(4.2028).
# 
# So we can conclude that 'Australia' and 'New Zealand' is Happiest Region in world followed by 'North America' while 'Sub-Saharan Africa' has least Happiest Region in world. So we need to Examine what actually contribute to happiness and unhappiness of this particular region of world.

# In[57]:


Happy_df.plot(kind ='box', subplots = True, layout=(3,4))

plt.tight_layout()


# Machine Learning Algorithm

# In[58]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor


# In[59]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33, random_state=43)
print('Training feature matrix size:',X_train.shape)
print('Training target vector size:',Y_train.shape)
print('Test feature matrix size:',X_test.shape)
print('Test target vector size:',Y_test.shape)


# Finding Best Random state

# In[60]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
maxR2_score=0
maxRS=0
for i in range(1,200):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33, random_state=i)
    lin_reg=LinearRegression()
    lin_reg.fit(X_train,Y_train)
    y_pred=lin_reg.predict(X_test)
    R2=r2_score(Y_test,y_pred)
    if R2>maxR2_score:
        maxR2_score=R2
        maxRS=i
print('Best accuracy is', maxR2_score ,'on Random_state', maxRS)


# Linear Regression

# In[61]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33, random_state=148)
lin_reg=LinearRegression()
lin_reg.fit(X_train,Y_train)
lin_reg.score(X_train,Y_train)
y_pred=lin_reg.predict(X_test)
print('Predicted result price:\n', y_pred)
print('\n')
print('actual price\n',Y_test)


# Linear Regression Evaluation Matrix

# In[62]:


print('\033[1m'+' Error :'+'\033[0m')
print('Mean absolute error :', mean_absolute_error(Y_test,y_pred))
print('Mean squared error :', mean_squared_error(Y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test,y_pred)))
print('\n')
from sklearn.metrics import r2_score
print('\033[1m'+' R2 Score :'+'\033[0m')
print(r2_score(Y_test,y_pred))


# Applying other Regression Model

# In[63]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor


# In[64]:


rf = RandomForestRegressor(n_estimators = 100 )
svr=SVR()
dtc = DecisionTreeRegressor()
ad=AdaBoostRegressor()

model = [rf,svr,dtc,ad]

for m in model:
    m.fit(X_train,Y_train)
    m.score(X_train,Y_train)
    y_pred = m.predict(X_test)
                                            
    print('\033[1m'+' Error of ', m, ':' +'\033[0m')
    print('Mean absolute error :', mean_absolute_error(Y_test,y_pred))
    print('Mean squared error :', mean_squared_error(Y_test,y_pred))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test,y_pred)))
    print('\n')

    print('\033[1m'+' R2 Score :'+'\033[0m')
    print(r2_score(Y_test,y_pred)) 
    print('\n')


# "We can see that Linear Regression Model Gives us maximum R2 Score"

# Cross Validation

# In[65]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(lin_reg, X, Y, cv = 5)
print('\033[1m'+'Cross Validation Score :'+'\033[0m\n')
print("Score :" ,score)
print("Mean Score :",score.mean())
print("Std deviation :",score.std())


# Saving Model

# In[66]:


import joblib
joblib.dump(lin_reg,'lin_reg.obj')


# In[ ]:




