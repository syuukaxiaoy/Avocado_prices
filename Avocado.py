#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score,explained_variance_score
from sklearn.model_selection import GridSearchCV

get_ipython().run_line_magic('matplotlib', 'inline')


# #### Recently fascinated with avocado dishes such as avocado sushi, milkshakes, sandwiches and more. So this avocado's data is of great interest to me, let's take a look at what's new.

# ### I will focus on three questions below. 
# - What are the trends in avocado prices in recent years? 
# - What are people's preferences for choice？
# - Was the Avocadopocalypse of 2017 real?

# In[2]:


#import data
avo = pd.read_csv("avocado.csv")


# In[3]:


#view data head
avo.head()


# In[4]:


#check data
avo.describe()


# In[5]:


#check info
avo.info()


# In[6]:


avo.columns


# In[7]:


#Data cleaning
#drop unnecessary columns
avo = avo.drop(['Unnamed: 0'],axis=1)


# In[8]:


#Date to datatime （type）
avo['Date'] = pd.to_datetime(avo['Date'])


# In[9]:


#check data
avo.head()


# In[10]:


avo.info()


# In[11]:


#check year
avo['year'].value_counts()


# In[12]:


#check the column of type
avo['type'].value_counts()


# In[13]:


#check region
avo['region'].value_counts()


# In[14]:


#Total of region
avo['region'].nunique()


# # Data Explorary

# The price is always important. Of course, the more expensive, the better, but the buyer may not be recognized. Let's take a look at the price distribution of our favorite avocados!

# In[15]:


#plot the distribution of averageprice
plt.figure(figsize=(10,8))
plt.title('Describe of averageprice')
ax = sns.distplot(avo['AveragePrice'],bins=50,kde=True)


# From this histogram we can see that this is a bimodal one. There are two kinds of avocado organic and conventional in the data, so I think this is the reason for the double peak.

# In[16]:


#Averageprice scatter plot for different types of avocados
plt.figure(figsize=(10,8))
plt.title('Averageprice vs Total Volume of different types')
ax =sns.scatterplot(x='Total Volume', y='AveragePrice', hue='type',data=avo)


# In this figure, we can see that the price distribution of different kinds of avocados is very different, and the price of organic avocados is generally higher than that of conventional ones.

# In[17]:


#set plot
plt.title('Averageprice of Type')
ax =  sns.boxplot(x='type', y='AveragePrice', data=avo)


# Because organic avocados cost more to grow, they are obviously more expensive. Some people like natural products and are willing to pay a higher price for them. But the price of avocado probably depends not only on the type. Let's take a look at the average price of avocado in different regions.

# In[18]:


#groupyby region
byregion = avo.groupby('region').mean()
byregion = byregion.reset_index()


# In[19]:


byregion.head()


# In[20]:


byregion.describe()


# In[21]:


#sort and plot the averageprice of different region
byregiona = byregion.sort_values(by='AveragePrice', ascending=False)
plt.figure(figsize=(16,10))
plt.title('Region vs Averageprice')
ax = sns.barplot(x='region', y='AveragePrice',data=byregiona)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-90);


# According to the average price comparison, the highest region is HartfordSpringfield about 1.82 a single avocado, the lowest is Houston about 1.05 and the averageprice is 1.41.Then let's look at the sales in these areas.

# In[22]:


#Of course the totalus is the most，we have to remove it first.
byregion['Total Volume'] = byregion['Total Volume']/1000
byregions = byregion.sort_values(by='Total Volume', ascending=False)
byregions = byregions.reset_index().drop(['index'],axis=1)
byregions = byregions.drop([0])
plt.figure(figsize=(16,10))
plt.title('Region vs Total Volume')
ax = sns.barplot(x='region', y='Total Volume',data=byregions)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-90);


# Of course the totalus is the most about more than 17 thousands，so we have to remove it at first.And we can see the highest 3 regions are West, Califormia, and SouthCentral.

# In[23]:


#boxplot of Year vs average Averageprice
plt.title('Year vs average Averageprice')
ax = sns.boxplot(x='year', y='AveragePrice', data=avo)


# The average price is highest in 2017, but interestingly, the highest price is in 2016.
# Let's take a look at the trends.

# In[24]:


import datetime


# In[25]:


#Extract month
avo['month'] = [i.month for i in avo['Date']]


# In[26]:


avomo = avo[['Date', 'AveragePrice', 'year']]
avomo['month'] = [i.month for i in avomo['Date']]


# In[27]:


avomo.head()


# In[28]:


plt.figure(figsize=(12,8))
plt.title('Averageprice trend of each year')
ax = sns.lineplot(x='month', y='AveragePrice', hue='year',markers=True, data=avo);


# Surely organic avocados are good? I'm sorry I haven't felt the difference yet, but how many organic avocados are good for you? I'm just like most people who are more attuned to taste, which is when the avocado is ripe, when it's at its best, or when it's committed to making something delicious. Maybe I'll just be a foodie. Do you like avocados as much as I do? Look forward to sharing more avocado recipes with me~

# In[29]:


#Averageprice vs type all year
plt.figure(figsize=(12,8))
plt.title('Averageprice vs type')
ax =  sns.boxplot(x='month', y='AveragePrice',hue='type', data=avo)


# That's right. Organic avocados have always been expensive。。。

# In[30]:


#heatmap of corr
plt.figure(figsize=(16,10))
ax = sns.heatmap(avo.corr(), annot=True, fmt=".2f");


# In[31]:


#set pairplot
ax = sns.pairplot(avo, palette="husl")


# In[32]:


#groupby month
bymon = avo.groupby(['month']).mean()
bymon = bymon.reset_index()


# In[33]:


#check bymon
bymon.head()


# In[34]:


#set lineplot of  bag size
x=bymon['month']
y1=bymon['Small Bags']
y2=bymon['Large Bags']
y3=bymon['XLarge Bags']
plt.figure(figsize=(10,8))
plt.title('Prference of bag size')
plt.plot(x, y1, color='green', label='S Bags')
plt.plot(x, y2, color='red', label='L Bags')
plt.plot(x, y3,  color='skyblue', label='XL Bags')
plt.legend() 
plt.xlabel('month')
plt.ylabel('Bags')
plt.show()


# The person who chooses Ssize bag is much higher than L, or XL. That is to say, people are more inclined to buy a small amount of avocados every time they buy. I think avocado is a banana-like food that has the characteristic of decaying once it matures. It is usually not mature in the counter of the supermarket. It is good to eat a little for a few days. If you buy mature avocados, you must eat them immediately. Otherwise, if you buy too much, they may mature at the same time but you can't finish them. That's too. . .So I am also buying avocados in small quantities as many people do.

# In[35]:


plt.figure(figsize=(12,8))
plt.title('Volume trend of each year')
ax = sns.lineplot(x='month', y='Total Volume', hue='year', data=avo);


# From the point of view of sales, the avocados are popular with people's tables. There was even a shortage in 2017 from google.why? Taste? health? Or appearance? Perhaps it is people's curiosity about new things. For example, the Super Bowl has developed avocado-related products, even if the goddess Miranda Kerco has contributed specifically to this... Then, the marketer set a position for the avocado - the luxury in the fruit.
# 
# No matter what age, luxury goods are always hot. The same is true for avocados. It is the flag of health, petty bourgeoisie and weight loss... The luxury goods in fruits are getting louder and louder, and the price of avocado is rising!

# In[36]:


#set groupby date
bydate = avo.groupby('Date').mean()
bydate = bydate.reset_index()
#check bydate
bydate.head()


# In[37]:


#plot lineplot
plt.figure(figsize=(20,10))
plt.title("The trends of avocado prices recent years")
ax = sns.pointplot(x='Date', y='AveragePrice',data=bydate)
ax.set_xticklabels(ax.get_xticklabels(), rotation=-90);


# There is an illusion in the world that avocados can only be afforded by the upper class, and it is also a nutritious food that best matches the identity of the upper class.Of course, avocado does have a lot of nutrients, but the truth is far from exaggerated, which means that the success of avocado is more from propaganda and hype.
# 
# So give everyone a wake up, even if there is something more nutritious, you need to eat rationally, don't blindly listen to the propaganda.... Rational consumption is actually applicable to all foods, you can eat, but eat in moderation, I believe everyone should always be at heart.

# # Data Preparation

# In[38]:


#convert type into dummies
dum_type = pd.get_dummies(avo['type'])
avo = pd.concat([avo, dum_type], axis=1)


# In[39]:


#convert region into codes
avo['region'] = avo['region'].astype('category')
avo['region'] = avo['region'].cat.codes


# # bulid model

# In[40]:


#bulid model

X_columns = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'year',
             'conventional', 'organic','region', 'month']

X = avo[X_columns]
y = avo['AveragePrice']


# In[41]:


# check X , Y shape
print(X.shape,y.shape)


# In[42]:


avo.head()


# In[43]:


#Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)


# In[44]:


#set lm mod
lm_model = LinearRegression(normalize=True) 
lm_model.fit(X_train, y_train) 

# Predictions
y_test_preds = lm_model.predict(X_test)
#Rsquared
r2_test = r2_score(y_test, y_test_preds)

#Print r2
r2_test


# In[45]:


#set xgboost model
mod = xgb.XGBRegressor()
mod.fit(X_train, y_train)
y_test_preds = mod.predict(X_test)

y_train_pred = mod.predict(X_train)
y_test_pred = mod.predict(X_test)

#print MSE and R^2
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


# In[46]:


#plot results
plt.figure(figsize = (10, 7))
plt.scatter(y_train_pred, y_train_pred - y_train, c = 'black', marker = 'o', s = 25, alpha = 0.5, label = 'Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c = 'lightgreen', marker = 's', s = 25, alpha = 0.3, label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'red')
plt.xlim([-2, 6])
plt.show()


# In[47]:


#Cross-validation and search best_params
params = {'max_depth': [3, 5, 10], 'learning_rate': [0.05, 0.1], 'subsample': [0.8, 0.85, 0.9], 'colsample_bytree': [0.5, 1.0]}

mod = xgb.XGBRegressor()
clf = GridSearchCV(mod, params, cv = 5, n_jobs = 1)
clf.fit(X_train, y_train)


# In[48]:


#print result
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


# In[49]:


#print best score and best params
print(clf.best_score_)
print(clf.best_params_)


# In[50]:


#plot result
plt.figure(figsize = (10, 7))
plt.scatter(y_train_pred, y_train_pred - y_train, c = 'black', marker = 'o', s = 5, alpha = 0.5, label = 'Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c = 'lightgreen', marker = 's', s = 5, alpha = 0.5, label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'red')
plt.xlim([-2, 6])
plt.show()


# In[ ]:




