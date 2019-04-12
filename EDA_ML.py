
# coding: utf-8

# # Load Data

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import gc
import os
import logging
import datetime
import warnings
import seaborn as sns
from tqdm import tqdm_notebook
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')
train=pd.read_csv('dataset/train.csv')
test=pd.read_csv('dataset/test.csv')
features = [c for c in train.columns if c not in ['ID_code', 'target']]


# # Take a look of data

# In[2]:


train.shape, test.shape


# In[3]:


train.head()


# # Checking Missing

# In[4]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[5]:


missing_data(train)


# In[7]:


missing_data(test)


# # Check the describe

# In[8]:


train.describe()


# In[9]:


test.describe()


# # Balance or Imbalance

# In[10]:


sns.countplot(train['target'], palette='Set3')


# # Density plots of the features

# In[11]:


def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


# In[13]:


t0 = train.loc[train['target'] == 0]
t1 = train.loc[train['target'] == 1]
features = train.columns.values[2:102]
plot_feature_distribution(t0, t1, '0', '1', features)


# In[14]:


features = train.columns.values[102:202]
plot_feature_distribution(t0, t1, '0', '1', features)


# #  Features corr

# In[15]:


correlations = train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.head(10)#TOP 10


# In[16]:


correlations.tail(10)#Last 10


# # Duplicate value

# In[17]:


features = train.columns.values[2:202]
unique_max_train = []
unique_max_test = []
for feature in features:
    values = train[feature].value_counts()
    unique_max_train.append([feature, values.max(), values.idxmax()])
    values = test[feature].value_counts()
    unique_max_test.append([feature, values.max(), values.idxmax()])


# In[19]:


#Train
np.transpose((pd.DataFrame(unique_max_train, columns=['Feature', 'Max duplicates', 'Value'])).            sort_values(by = 'Max duplicates', ascending=False).head(15))


# In[18]:


#Test
np.transpose((pd.DataFrame(unique_max_test, columns=['Feature', 'Max duplicates', 'Value'])).            sort_values(by = 'Max duplicates', ascending=False).head(15))


# # Density plots of the features counts

# In[20]:


train_fre=pd.read_csv('train_fre.csv')


# In[22]:


fre_features=[c for c in train_fre.columns if 'fre_' in c]
t0_fre=train_fre[train_fre['target']==0][fre_features]
t1_fre=train_fre[train_fre['target']==1][fre_features]


# In[28]:


plot_feature_distribution(t0_fre, t1_fre, '0', '1', fre_features[0:100])


# In[29]:


plot_feature_distribution(t0_fre, t1_fre, '0', '1', fre_features[100:])


# # Feature Counts

# In[2]:


merge=pd.concat([train[features],test[features]],axis=0)


# In[3]:


merge.shape


# In[5]:


merge_fre=pd.DataFrame()
for f in features:
    print(f)
    uni_=merge[f].unique().round(4)
    dic_var=dict.fromkeys(uni_,0)
    merge_np=np.array(merge[f]).round(4)
    fre_=[]
    c=merge[f].round(4).value_counts()
    for uni in uni_:
        dic_var[uni]=c.loc[uni]
    for i in merge_np:
        fre_.append(dic_var[i])
    merge_fre['fre_'+f]=fre_


# In[6]:


merge_fre.to_csv('merge_fre.csv',index=False)

