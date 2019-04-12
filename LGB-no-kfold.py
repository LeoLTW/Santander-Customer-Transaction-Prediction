
# coding: utf-8

# In[9]:


import numpy as np # linear algebra
from scipy.stats import norm
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
target = train['target']
features = [c for c in train.columns if c not in ['ID_code', 'target']]


# In[10]:


merge_f=pd.read_csv('merge_fre.csv')
train_all=pd.concat([train[features],test[features]],axis=0)
train_all=train_all.reset_index(drop=True)
train_f_all=pd.DataFrame()
for f in features:
    train_f_all['me_'+f]=train_all[f]*merge_f['fre_'+f]


# In[13]:


train_f_all.shape


# In[14]:


le = LabelEncoder()
ff=[c for c in train_f_all.columns if 'me_' in c]
for f in ff:
    train_f_all[f]=le.fit_transform(train_f_all[f])


# In[ ]:


train_f_all.shape


# In[16]:


features = [c for c in train.columns if c not in ['ID_code', 'target']]
train_new=train[features]
train_new=pd.concat([train_new,train_f_all[:200000]],axis=1)


# In[17]:


train_new.shape


# In[18]:


train_new


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(train_new,target,test_size=0.2,random_state=627,shuffle=True)


# In[20]:


param = {
    'bagging_freq': 10, #handling overfitting
    'bagging_fraction': 0.2,#handling overfitting - adding some noise
     #'boost': 'dart', 
    #'boost': 'goss',
     'boost_from_average':False,
     'boost': 'gbdt',   
    'feature_fraction': 0.15, #handling overfitting
    'learning_rate': 0.01, #the changes between one auc and a better one gets really small thus a small learning rate performs better
    'max_depth':2, 
    'metric':'auc',
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'xentropy', 
    'verbosity':1,
    "bagging_seed" : 122,
    "seed": 20,
    }


# In[21]:


oof = np.zeros(len(train))

trn_data = lgb.Dataset(x_train, label=y_train)
val_data = lgb.Dataset(x_test, label=y_test)
    
clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)
oof = clf.predict(train_new, num_iteration=clf.best_iteration)

print("roc score: {:<8.5f}".format(roc_auc_score(target, oof)))


# In[552]:


test_fre=merge_f[200000:]
test_fre=test_fre.reset_index(drop=True)


# In[553]:


test_fre


# In[554]:


test_f=pd.DataFrame()
for f in features:
    test_f['me_'+f]=test[f]*test_fre['fre_'+f]

le = LabelEncoder()
ff=[c for c in test_f.columns]
for f in ff:
    test_f[f]=le.fit_transform(test_f[f])

features = [c for c in train.columns if c not in ['ID_code', 'target']]
test_new=test[features]
test_new=pd.concat([test_new,test_f],axis=1)


# In[556]:


test_new


# In[557]:


predictions = clf.predict(test_new, num_iteration=clf.best_iteration)


# In[558]:


sub = pd.DataFrame({"ID_code": test.ID_code.values})
sub["target"] = predictions
sub.to_csv("submission_lgb_final_1.csv", index=False)

